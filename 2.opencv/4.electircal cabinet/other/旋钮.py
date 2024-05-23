import torch
from torchvision import transforms
import cv2
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR, draw_ocr

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = torch.hub.load('ultralytics/yolov5',  'custom', path=model_path, source='local')
model = torch.hub.load(r'F:\PythonWork\1.detection\1.yolov5-master/', 'custom', source='local', path=r'F:\PythonWork\1.detection\1.yolov5-master\runs\train-cls\exp2\weights\best.pt')
model.to(device)

# 加载图片
image_path = r'F:\PythonWork\3.opencv\4.electircal cabinet\img_test\button\button2-2.jpg'
image = Image.open(image_path)

# 转换图像为合适的张量格式
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])
image_tensor = transform(image).unsqueeze(0)  # 添加批量维度
image_tensor = image_tensor.to(device)  # 将张量移到同一设备上

# 进行检测
with torch.no_grad():
    results = model(image_tensor)

pred = F.softmax(results, dim=1)

for i, prob in enumerate(pred):
    top5i = prob.argsort(0, descending=True)[:5].tolist()
    print(model.names[top5i[0]])
    text = '\n'.join(f'{prob[j]:.2f} {model.names[j]}' for j in top5i)
    print(text)


ocr = PaddleOCR(use_angle_cls=True, lang="ch")
ocr_result = ocr.ocr(image_path)
print(ocr_result)


result = ocr_result[0]
image = Image.open(image_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='doc/fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')






