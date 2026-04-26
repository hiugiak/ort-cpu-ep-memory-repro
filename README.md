# ONNX Runtime CPU EP Memory Repro

Minimal Rust repro for sustained dirty/RSS memory growth observed on Apple
Silicon when repeatedly running a common object detection model through ONNX
Runtime's CPU execution provider with fresh changing inputs.

The app intentionally avoids application-specific capture, OpenCV, preview, and
Tauri code. It only:

- dynamically loads `libonnxruntime.dylib`
- registers the CPU execution provider
- loads a public YOLOv8n ONNX object detection model
- generates a fresh random `[1, 3, 640, 640]` `f32` tensor each iteration
- extracts the first output tensor so the run is fully materialized
- prints process RSS periodically

## Model

Download a public YOLOv8n ONNX model with:

```sh
./scripts/download-yolov8n.sh
```

The script downloads:

```text
https://huggingface.co/webml/yolov8n/resolve/main/onnx/yolov8n.onnx
```

## Run

From this directory:

```sh
cargo run --release -- \
  /path/to/libonnxruntime.dylib \
  models/yolov8n.onnx
```

Example using a local ONNX Runtime dylib:

```sh
cargo run --release -- \
  libs/libonnxruntime.dylib \
  models/yolov8n.onnx
```

Optional arguments:

```text
ort-cpu-ep-memory-repro <libonnxruntime.dylib> <model.onnx> [intra_threads] [print_every]
```

Defaults:

- `intra_threads`: `1`
- `print_every`: `100`

Example printing every 20 iterations:

```sh
cargo run --release -- \
  libs/libonnxruntime.dylib \
  models/yolov8n.onnx \
  1 \
  20
```

## Expected Observation

On the affected machine, Instruments showed dirty memory and `MALLOC_SMALL`
growing continuously during the CPU EP run. The allocation call tree pointed to:

```text
OrtApis::Run
onnxruntime::InferenceSession::Run
onnxruntime::Conv<float>::Compute
MlasConv
ArmKleidiAI::MlasConv
operator new
_malloc_type_malloc_outlined
```

The application-specific workload stopped showing the growth after switching to
CoreML Execution Provider first, with CPU as fallback. This repro keeps CPU EP
only so the CPU/MLAS/KleidiAI path can be inspected directly.
