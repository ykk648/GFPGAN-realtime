import cv2
import numpy as np
from tqdm import tqdm
import os
import time
from cv2box import CVImage, CVVideoLoader, get_path_by_ext
from cv2box.cv_gears import Linker, Consumer, CVVideoThread, Queue, CVVideoWriterThread, CVCamThread

from face_lib.face_restore import GFPGAN
from face_lib.face_detect_and_align import FaceDetect5Landmarks


def save_video_ffmpeg(input_video_path, output_video_path):
    audio_file_path = input_video_path.replace('.mp4', '.aac')
    if not os.path.exists(audio_file_path):
        os.system(
            'ffmpeg -y -hide_banner -loglevel error -i "'
            + str(input_video_path)
            + '" -f wav -vn  "'
            + str(audio_file_path)
            + '"'
        )
    if os.path.exists(audio_file_path):
        os.rename(output_video_path, output_video_path.replace('.mp4', '_no_audio.mp4'))
        start = time.time()
        os.system(
            'ffmpeg -y -hide_banner -loglevel error  -i "'
            + str(output_video_path.replace('.mp4', '_no_audio.mp4'))
            + '" -i "'
            + str(audio_file_path)
            + '" -c:v libx264 "'
            + str(output_video_path)
            + '"'
        )
        print('add audio time cost', time.time() - start)
        os.remove(output_video_path.replace('.mp4', '_no_audio.mp4'))
        os.remove(audio_file_path)
    return output_video_path


class FaceDetectThread(Linker):
    def __init__(self, queue_list: list):
        super().__init__(queue_list, fps_counter=True)
        self.fd = FaceDetect5Landmarks(mode='scrfd_500m')

    def forward_func(self, something_in):
        frame = something_in[0]
        bboxes_scrfd, kpss_scrfd = self.fd.get_bboxes(frame, min_bbox_size=64)
        if len(bboxes_scrfd) == 0:
            return [frame, None, None, None]

        face_image_, mat_rev_, roi_box_ = self.fd.get_single_face(crop_size=512, mode='mtcnn_512',
                                                                  apply_roi=True)
        return [frame, face_image_, mat_rev_, roi_box_]


class FaceRestoreThread(Linker):
    def __init__(self, queue_list: list):
        super().__init__(queue_list, fps_counter=True)
        self.gfp = GFPGAN(model_type='GFPGANv1.4', provider='gpu')

    def forward_func(self, something_in):
        src_face_image_ = something_in[1]
        if src_face_image_ is None:
            return [None] + something_in
        face_restore_out_ = self.gfp.forward(src_face_image_)
        CVImage(src_face_image_).show(1, 'before')
        CVImage(face_restore_out_).show(1, 'after')
        return [face_restore_out_] + something_in


class FaceParseThread(Linker):
    def __init__(self, queue_list: list):
        super().__init__(queue_list, fps_counter=True)

        self.face_mask_ = self.get_face_mask(mask_shape=(512, 512))

    def get_face_mask(self, mask_shape):
        mask = np.zeros((512, 512)).astype(np.float32)
        cv2.ellipse(mask, (256, 256), (220, 160), 90, 0, 360, (255, 255, 255), -1)
        thres = 20
        mask[:thres, :] = 0
        mask[-thres:, :] = 0
        mask[:, :thres] = 0
        mask[:, -thres:] = 0
        # Soften the mask edges before compositing.
        mask = cv2.stackBlur(mask, (201, 201))

        mask = mask / 255.
        mask = cv2.resize(mask, mask_shape)
        return mask[..., np.newaxis]

    def forward_func(self, something_in):
        if something_in[0] is None:
            return something_in + [None]

        return something_in + [self.face_mask_]


class FaceReverseThread(Consumer):
    def __init__(self, queue_list: list):
        super().__init__(queue_list, fps_counter=True)
        self.counter = 0
        self.start_time = time.time()

    def forward_func(self, something_in):
        face_restore_out = something_in[0]
        src_img_in = something_in[1]
        if face_restore_out is not None:
            mat_rev = something_in[3]
            roi_box = something_in[4]
            face_mask_ = something_in[5]
            restore_roi = CVImage(None).recover_from_reverse_matrix(face_restore_out,
                                                                    src_img_in[roi_box[1]:roi_box[3],
                                                                    roi_box[0]:roi_box[2]],
                                                                    mat_rev, img_fg_mask=face_mask_)


            src_img_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = restore_roi


if __name__ == '__main__':
    q0 = Queue(2)
    q1 = Queue(2)
    q2 = Queue(2)
    q3 = Queue(2)
    q4 = Queue(2)

    cvvl = CVCamThread(0, [q0])
    fdt = FaceDetectThread([q0, q1])
    frt = FaceRestoreThread([q1, q2])
    fpt = FaceParseThread([q2, q3])
    fret = FaceReverseThread([q3, q4])

    threads_list = [cvvl, fdt, frt, fpt, fret]

    for thread_ in threads_list:
        thread_.start()

    for thread_ in threads_list:
        thread_.join()
