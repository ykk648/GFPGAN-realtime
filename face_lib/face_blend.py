"""Optional GPU implementation of face warping and mask compositing."""

import numpy as np


def blend_face_cvcuda(restored_face, background_roi, face_to_roi, face_mask, device="cuda"):
    """Warp a restored face into an ROI and blend it using CV-CUDA.

    Args:
        restored_face: HWC BGR float image with values in [0, 1].
        background_roi: HWC BGR uint8 background image.
        face_to_roi: A 2x3 affine matrix mapping face pixels to ROI pixels.
        face_mask: HW or HW1 float mask in [0, 1], matching the face size.
        device: PyTorch CUDA device used by the optional backend.

    Returns:
        A uint8 BGR NumPy image with the same shape as background_roi.

    PyTorch and CV-CUDA are imported only when this function is called.
    The default video and camera pipelines use the existing cv2box backend.
    """
    face = np.asarray(restored_face, dtype=np.float32)
    background = np.asarray(background_roi)
    matrix = np.asarray(face_to_roi, dtype=np.float32)
    mask = np.asarray(face_mask, dtype=np.float32)
    if face.ndim != 3 or face.shape[2] != 3 or 0 in face.shape:
        raise ValueError("restored_face must be a nonempty HWC image with 3 channels")
    if background.ndim != 3 or background.shape[2] != 3 or 0 in background.shape:
        raise ValueError("background_roi must be a nonempty HWC image with 3 channels")
    if background.dtype != np.uint8:
        raise ValueError("background_roi must have dtype uint8")
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("face_to_roi must be a finite 2x3 affine matrix")
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]
    if mask.shape != (*face.shape[:2], 1):
        raise ValueError("face_mask must have shape HW or HW1 matching restored_face")
    for name, array in [("restored_face", face), ("face_mask", mask)]:
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
            raise ValueError(f"{name} must contain finite values in [0, 1]")

    try:
        import cvcuda
        import torch
    except ImportError as exc:
        raise RuntimeError("This optional backend requires PyTorch and CV-CUDA") from exc

    cuda_device = torch.device(device)
    if cuda_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CV-CUDA face blending requires an available CUDA device")

    with torch.cuda.device(cuda_device):
        face_gpu = torch.as_tensor(np.ascontiguousarray(face), device=cuda_device).unsqueeze(0)
        mask_gpu = torch.as_tensor(np.ascontiguousarray(mask), device=cuda_device).unsqueeze(0)
        background_gpu = torch.as_tensor(
            np.ascontiguousarray(background), dtype=torch.float32, device=cuda_device
        ).unsqueeze(0)
        height, width = background.shape[:2]
        warped_face = torch.empty((1, height, width, 3), dtype=torch.float32, device=cuda_device)
        warped_mask = torch.empty((1, height, width, 1), dtype=torch.float32, device=cuda_device)

        # Synchronize at the PyTorch/CV-CUDA stream boundaries.
        torch.cuda.synchronize(cuda_device)
        for source, destination in [(face_gpu, warped_face), (mask_gpu, warped_mask)]:
            cvcuda.warp_affine_into(
                dst=cvcuda.as_tensor(destination, "NHWC"),
                src=cvcuda.as_tensor(source, "NHWC"),
                xform=np.ascontiguousarray(matrix),
                flags=cvcuda.Interp.LINEAR,
                border_mode=cvcuda.Border.CONSTANT,
                border_value=[0],
            )
        torch.cuda.synchronize(cuda_device)
        alpha = warped_mask.clamp(0, 1)
        blended = warped_face * (255.0 * alpha) + background_gpu * (1.0 - alpha)
        return blended.clamp(0, 255).to(torch.uint8).squeeze(0).cpu().numpy()
