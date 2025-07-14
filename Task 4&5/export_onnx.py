import torch

# load your model - example with YOLOv5 from ultralytics repo:
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
dummy_input = torch.randn(1, 3, 640, 640)

torch.onnx.export(model, dummy_input, "yolov5s.onnx", opset_version=12)
print("Export completed.")
