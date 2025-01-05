# OpenMPF Puzzle Piece Detection

OpenMPF component for detecting edge and regular puzzle pieces using a fine-tuned YOLOv11 model

## Job Properties

`CONFIDENCE`: The minimum confidence for a detection to be included. Defaults to 0.7
`INCLUDE_EDGE`: "true" or "false", signaling whether edge puzzle piece detections should be included. Defaults to "true"
`INCLUDE_REGULAR`: "true" or "false", signaling whether regular puzzle piece detections should be included. Defaults to "true"

## Running with NVIDIA GPUs

The Ultralytics implementation of YOLO is capable of performing inference on NVIDIA GPUs and even auto-detects and uses attached devices. To get this working with Docker, simply run the container with the NVIDIA container runtime and set the `NVIDIA_VISIBLE_DEVICES` environment variable as described (here)[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.10.0/user-guide.html#gpu-enumeration].
