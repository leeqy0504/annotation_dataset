# 标注数据集流水线（Annotation Dataset Pipeline）

一个仅提供 CLI（命令行）接口的工具包，用于根据 RGB 图像序列或视频输入和 SAM2 掩码生成 **COCO 格式目标检测与实例分割数据集**。

## 流水线执行内容

默认的 `annotation_dataset` 流水线定义在：

`configs/pipelines/annotation_dataset.yaml`

执行流程如下：

1. `prompt_mask` —— 使用 SAM2 点提示（point prompts）生成第一帧掩码。
2. `sam2_video_propagation` —— 将掩码传播到整个 RGB 视频帧序列。
3. `mask_qa` —— 计算掩码对应的检测框，并标记可疑帧。
4. `review_pack` —— 生成本地 HTML 审核包。
5. `detection_dataset_export` —— 导出 UniTrain 推荐的 COCO 数据集、复制图片、导出掩码以及预览图。

整个流程无需 Web 服务或前端界面。

## 目录结构

```text
annotation_dataset/
  configs/
    pipelines/annotation_dataset.yaml
    algorithms/sam2.yaml
    runtime/server.yaml
  pipeline/
  registry/classes.json
  tasks/<task_name>/
    task.yaml
    dataset_info.json
    rgb/*.png
    source.mp4
  tools/sam2/
  run_annotation_dataset.sh
```

配置加载器采用分层架构：

`tasks/<task>/task.yaml` 中引用 `pipeline: annotation_dataset` 和
`runtime: server`；这些名称会分别解析到 `configs/pipelines/` 和
`configs/runtime/` 下对应的配置文件，然后再合并
`configs/algorithms/` 中定义的算法默认配置。

## 准备一个任务

创建 `tasks/<task_name>/task.yaml`：

```yaml
task_id: mouse_001
pipeline: annotation_dataset
runtime: server
class_id: 0
input:
  rgbd_dir: ./tasks/mouse_001/
  first_frame: 0
  # 可选：如果未提供 rgb/*.png，则从视频抽帧生成 RGB 图像序列
  video_path: ./tasks/mouse_001/source.mp4
  frame_interval: 1
sam2:
  points: [[380, 182]]
  labels: [1]
detection_dataset:
  class_name: object
  class_id: 0
  min_box_area: 16
  train_ratio: 0.8
output_dir: output/
```

方式一：将 RGB 图像放入：

```text
tasks/<task_name>/rgb/*.png
```

方式二：在 `task.yaml` 中配置视频输入：

```yaml
input:
  rgbd_dir: ./tasks/<task_name>/
  video_path: ./tasks/<task_name>/source.mp4
  frame_interval: 1
```

当 `rgb/` 中还没有 PNG 帧时，流水线会调用 `ffmpeg` 抽帧到：

```text
tasks/<task_name>/rgb/%06d.png
```

`frame_interval` 默认为 `1`，表示每帧都抽；设置为 `5` 表示每 5 帧抽 1 帧。

如果存在 `tasks/<task_name>/dataset_info.json`，并且其中包含：

* `sam2_points.points`
* `sam2_points.labels`

则这些点提示会覆盖 `task.yaml` 中对应的配置。

## 运行

```bash
pip install -e .
./run_annotation_dataset.sh --task mouse_001 --force
```

等价的直接运行命令：

```bash
python -m pipeline.cli run --config tasks/mouse_001/task.yaml --force
```

默认情况下，输出结果位于：

```text
output/<task_name>/
```

如果在任务配置中设置了 `run_id`，则输出目录变为：

```text
output/<task_name>/runs/<run_id>/stages/
```

## SAM2 Docker 挂载

`configs/algorithms/sam2.yaml` 默认假设项目在 SAM2 容器中的挂载路径为：

```text
/home/try/code/annotation_dataset
```

请确保 `sam2.container` 指定的 Docker 容器将项目目录挂载到上述相同路径；如果挂载路径不同，请修改配置中的 `sam2.project_mount`。

## UniTrain COCO 输出

导出目录使用 UniTrain 推荐的 COCO 数据集结构。导出前会先按连续帧切 clip，再按 clip 划分：

```text
clip_001: 连续帧片段
clip_002: 连续帧片段
clip_003: 连续帧片段
...
```

默认不需要配置 `clip_size`：导出器会根据总帧数自适应切成最多 10 个连续 clip，再按 `train_ratio: 0.8` 划分，即 80% clips 进入 `train/`，20% clips 进入 `valid/`。同一个 clip 内的连续帧不会被拆到不同 split。

如果你想固定每个 clip 的长度，也可以显式配置：

```yaml
detection_dataset:
  clip_size: 500
  train_ratio: 0.8
```

```text
output/<task_name>/detection_dataset_export/
  train/
    *.png
    _annotations.coco.json
  valid/
    *.png
    _annotations.coco.json
  masks/*.png
  preview/*.svg
  contact_sheet.svg
```

训练配置可指向 `detection_dataset_export/`：

```yaml
data:
  path: /path/to/output/<task_name>/detection_dataset_export
  format: coco
```

每个 split 的 `_annotations.coco.json` 至少包含 COCO 标准字段：

* `images`
* `annotations`
* `categories`

每个 annotation 包含 detection 与 segmentation 所需字段：

* `bbox`: COCO `[x, y, width, height]`
* `segmentation`: COCO uncompressed RLE，包含 `size` 和 `counts`
* `area`
* `iscrowd`: `0`

因此同一份导出可用于：

* `task: detect`：读取 `bbox`
* `task: segment`：读取 `segmentation`

## End-to-End UniTrain Pipeline

当一个任务需要先生成标注数据集，再用 UniTrain 训练模型时，使用 `pipeline: annotation_to_unitrain`。

任务配置示例：

```yaml
task_id: mouse_001
pipeline: annotation_to_unitrain
runtime: server
class_id: 0
input:
  rgbd_dir: ./tasks/mouse_001/
  video_path: ./tasks/mouse_001/source.mp4
  frame_interval: 1
sam2:
  points: [[380, 182]]
  labels: [1]
detection_dataset:
  class_name: object
  class_id: 0
  train_ratio: 0.8
training: rfdetr_seg_nano
training_overrides:
  train:
    epochs: 20
    batch: 4
    device: 0
output_dir: output/
```

运行：

注意：`model_train` 会调用 UniTrain runner，训练阶段需要提前准备匹配的框架依赖和可用设备/GPU；依赖或设备缺失会在训练阶段失败。

注意：pipeline 的 `model_train` 在 MVP 阶段目前仅支持 RF-DETR。独立 UniTrain 仍可能包含 YOLO/Ultralytics 工具，但 pipeline 内的 YOLO/Ultralytics 训练还需要后续补齐数据集转换桥接。

```bash
python -m pipeline.cli run --config tasks/mouse_001/task.yaml --force
```

未设置 `run_id` 时，最终结果位置：

```text
output/<task>/model_train/train_result.json
output/<task>/manifest.json
```

设置 `run_id` 时，最终结果位置：

```text
output/<task>/runs/<run_id>/stages/model_train/train_result.json
output/<task>/runs/<run_id>/manifest.json
```
