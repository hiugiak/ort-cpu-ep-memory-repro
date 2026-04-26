#!/usr/bin/env sh
set -eu

mkdir -p models
curl -L \
  --output models/yolov8n.onnx \
  https://huggingface.co/webml/yolov8n/resolve/main/onnx/yolov8n.onnx

echo "Downloaded models/yolov8n.onnx"
