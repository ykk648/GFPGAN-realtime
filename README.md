# GFPGAN Realtime · 实时人脸超分

通过多线程流水线，让 **GFPGAN 人脸修复以实时速度运行**。基于 ONNX Runtime GPU 推理，支持摄像头实时预览和本地视频批量处理。

本项目于 **2023 年实现**，现在将当时的实现整理并公开，采用非商用许可证发布。

这个仓库关注的是人脸超分的工程实现：将视频读取、人脸检测、修复、融合和输出拆到不同线程，通过有界队列连接，让视频 I/O 与推理阶段并行推进。适合用作实时摄像头画质增强、视频人脸修复，以及自定义人脸处理流水线的起点。

## 为什么用这个仓库

- **面向实时视频**：SCRFD 轻量人脸检测搭配 GFPGAN v1.4，在摄像头画面中持续修复人脸，并展示修复前后的脸部预览。
- **多线程流水线**：各处理阶段独立运行，队列限制在途帧数，避免把整段视频加载到内存。
- **局部修复与回贴**：将人脸对齐到 512 × 512，完成修复后，通过逆仿射变换和柔化遮罩融合回原图，保持原始画面尺寸。
- **便于改造**：检测、修复、融合分别封装在线程类中，可以替换模型或调整处理阶段；同一套组件也用于离线视频处理。

原项目曾在 **RTX 3080、1080p 视频上达到约 30 FPS**。这是历史测试参考，实际吞吐和延迟取决于 GPU、输入分辨率、人脸大小和编码方式；本次整理未重新进行性能测试。

## 工作流程

```text
摄像头 / 视频读取
       ↓
SCRFD 检测 → 五点对齐 → GFPGAN 修复 → 遮罩融合与回贴 → 预览 / 视频输出
```

每帧选择一张人脸修复，默认按检测置信度选择；未检测到人脸时透传原始帧。修复针对人脸区域，输出画面保持输入尺寸。

### 可选的 CV-CUDA 融合

[`face_lib/face_blend.py`](face_lib/face_blend.py) 提供独立的 `blend_face_cvcuda` 函数，将人脸仿射回贴和遮罩融合放到 GPU 上执行。默认流水线继续使用 cv2box 的融合实现，需要切换时可在 `FaceReverseThread.forward_func` 中调用该函数。

函数接收修复后的人脸、背景 ROI、人脸到 ROI 的仿射矩阵及人脸遮罩，返回融合后的 BGR 图像。它仅在调用时加载 PyTorch 和 CV-CUDA；使用该可选实现需要额外安装与 CUDA 环境匹配的 [PyTorch](https://pytorch.org/) 和 [CV-CUDA](https://github.com/CVCUDA/CV-CUDA)。

## 安装

建议使用 **Python 3.9 或 3.10**，以及 NVIDIA GPU 和与 ONNX Runtime 匹配的 CUDA / cuDNN 环境。当前依赖使用 `onnxruntime-gpu==1.13.1`，可参考 [ONNX Runtime CUDA 兼容表](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) 准备运行环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

所有命令均在仓库根目录执行。摄像头预览需要可用的摄像头和图形桌面环境。

## 下载模型

`pretrain_models/` 用于保存本地权重，整个目录已加入 `.gitignore`。运行默认的摄像头和视频流水线只需下面两个模型：

| 模型 | 公开下载地址 | 本地保存位置 |
| --- | --- | --- |
| SCRFD 500M，含五点关键点 | [InsightFace 官方 buffalo_sc.zip](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip)，取其中的 `det_500m.onnx` | `pretrain_models/face_lib/face_detect/scrfd_onnx/scrfd_500m_bnkps_shape640x640.onnx` |
| GFPGAN v1.4 ONNX | [FaceFusion 模型发布页中的 gfpgan_1.4.onnx](https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx) | `pretrain_models/face_lib/face_restore/gfpgan/GFPGANv1.4.onnx` |

可以直接下载并放到对应位置：

```bash
mkdir -p pretrain_models/face_lib/face_detect/scrfd_onnx
mkdir -p pretrain_models/face_lib/face_restore/gfpgan

curl -L --fail \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip \
  -o pretrain_models/buffalo_sc.zip

unzip -p pretrain_models/buffalo_sc.zip det_500m.onnx \
  > pretrain_models/face_lib/face_detect/scrfd_onnx/scrfd_500m_bnkps_shape640x640.onnx

curl -L --fail \
  https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx \
  -o pretrain_models/face_lib/face_restore/gfpgan/GFPGANv1.4.onnx
```

SCRFD 文件名沿用项目约定；官方 `det_500m.onnx` 支持动态输入，代码加载时使用 640 × 640。GFPGAN 使用输入和输出均为 `1 × 3 × 512 × 512` 的单输入 ONNX 模型。

`face_lib/face_parsing/` 还保留了可选的人脸解析模块，模型路径为 `pretrain_models/face_lib/face_parsing/79999_iter.onnx`。默认流水线使用几何遮罩，无需下载该模型；需要启用时，可参考 [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) 获取原始权重并转换为 ONNX。

## 摄像头实时人脸超分

```bash
python face_restore_gfpgan_thread_cam_pipeline.py
```

默认打开摄像头 `0`，显示 `before` 和 `after` 两个人脸窗口。需要更换摄像头时，修改脚本末尾的 `CVCamThread(0, [q0])` 参数。通过终端 `Ctrl+C` 结束运行。

## 批量处理视频

```bash
python batch.py ./videos
```

将目录中的 `.mp4` 视频逐个处理，结果写入原目录，文件名增加 `_SR` 后缀，例如 `clip.mp4` → `clip_SR.mp4`。默认输出不携带原始音轨。

```bash
python batch.py --help
```

## 代码结构

```text
batch.py                                    批量视频命令行入口
face_restore_gfpgan_thread_pipeline.py       视频处理流水线
face_restore_gfpgan_thread_cam_pipeline.py   摄像头实时演示
face_lib/
  face_blend.py                             可选的 CV-CUDA 回贴与融合函数
  face_detect_and_align/                    SCRFD 检测与人脸对齐
  face_restore/                             GFPGAN ONNX 推理
  face_parsing/                             可选人脸解析模块
model_lib/                                  ONNX Runtime 模型封装
pretrain_models/                            本地模型目录，不进入 Git
```

## 许可证

本项目的原创代码采用 [PolyForm Noncommercial 1.0.0](LICENSE) 许可证，以源码可用（source-available）方式发布，仅授权许可条款规定的非商业用途。

- 允许非商业目的的学习、研究、使用、修改和再分发。
- 再分发时须提供许可证正文或其链接，并保留 `Required Notice:` 声明。
- 本许可证不授予商业使用权，包括将本项目用于商业产品、收费服务或其他以商业利益为目的的使用；完整授权范围以 [LICENSE](LICENSE) 正文为准。

本许可证仅适用于项目原创代码。第三方代码、依赖和模型继续遵循各自上游许可，本许可证不会替代上游条款或授予额外权利。

## 上游项目与模型许可

- [GFPGAN](https://github.com/TencentARC/GFPGAN)：人脸修复模型。
- [InsightFace / SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)：人脸检测、关键点和相关实现，代码中保留了原作者信息。
- [GFPGAN-onnxruntime-demo](https://github.com/xuanandsix/GFPGAN-onnxruntime-demo)：GFPGAN ONNX 推理与导出参考。
- [FaceFusion 模型发布页](https://github.com/facefusion/facefusion-assets/releases/tag/models-3.0.0)：本文使用的 GFPGAN v1.4 ONNX 下载来源。
- [cv2box](https://github.com/ykk648/cv2box)：视频读取、多线程队列和图像处理组件。
- [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)：可选人脸解析模块的模型来源。

第三方代码和模型遵循各自上游的许可证与使用条件。InsightFace 官方预训练模型限定用于非商业研究，具体见其 [许可说明](https://github.com/deepinsight/insightface#license)。
