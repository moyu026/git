import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load(r'F:\PythonWork\1.detection\1.yolov5-master/', 'custom', source='local', path=r'F:\PythonWork\1.detection\1.yolov5-master\runs\light\exp\weights\best.pt')
model.to(device)

# Image
im = r'F:\PythonWork\3.opencv\4.electircal cabinet\img_test\9.jpg'

# Inference
results = model(im)

print(results.pandas().xyxy[0])

