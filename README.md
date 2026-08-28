# fisheye-detection-training

Trains a YOLO26-based person detector on fisheye imagery (WoodScape-based),
exports to ONNX. Inference, C++ deployment, and ROS2 integration happen in
separate repos.

## Scope
- Data: WoodScape pedestrian subset (+ small self-labeled CVAT slice)
- Model: YOLO26n/s via Ultralytics
- Output: trained weights + ONNX export, versioned and tracked
EOF