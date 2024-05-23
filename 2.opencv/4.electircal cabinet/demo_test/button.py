from paddleocr import PaddleOCR, draw_ocr
import torch
from torchvision import transforms
import cv2
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

ocr = PaddleOCR(use_angle_cls=True, lang="ch")

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = torch.hub.load('ultralytics/yolov5',  'custom', path=model_path, source='local')
model = torch.hub.load(r'F:\PythonWork\1.detection\1.yolov5-master/', 'custom', source='local', path=r'F:\PythonWork\1.detection\1.yolov5-master\runs\train-cls\exp2\weights\best.pt')
model.to(device)

# 加载图片
image_path = r'button1.jpg'
# image = Image.open(image_path)
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

button_name = ['10KV117开关手车操作开关', '10KV117开关分/合闸转换开关', '10KV11745接地刀闸操作开关', '远方/就地转换开关']
button_state_name = ['left', 'middle', 'right']
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
    button_state = model.names[top5i[0]]
    # print(model.names[top5i[0]])
    # text = '\n'.join(f'{prob[j]:.2f} {model.names[j]}' for j in top5i)
    # print(text)




ocr_result = ocr.ocr(image)
# with open('button_name.txt', 'a') as f:
#     for idx in range(len(ocr_result)):
#         res = ocr_result[idx]
#         for line in res:
#             print(line)
            # if line[1][0] in button_name:
            #     f.write(str(line[1][0]) + '\n')


for idx in range(len(ocr_result)):
    res = ocr_result[idx]
    for line in res:
        if line[1][0] == button_name[0]:
            if button_state == button_state_name[0]:
                img = cv2ImgAddText(image, f'{button_name[0]}：至实验位', 10, 65, (255, 0, 0), 50)
            elif button_state == button_state_name[1]:
                img = cv2ImgAddText(image, f'{button_name[0]}：正常', 10, 65, (255, 0, 0), 50)
            elif button_state == button_state_name[2]:
                img = cv2ImgAddText(image, f'{button_name[0]}：至工作位', 10, 65, (255, 0, 0), 50)

        if line[1][0] == button_name[1]:
            if button_state == button_state_name[0]:
                img = cv2ImgAddText(image, f'{button_name[1]}：分闸', 10, 65, (255, 0, 0), 50)
            elif button_state == button_state_name[1]:
                img = cv2ImgAddText(image, f'{button_name[1]}：正常', 10, 65, (255, 0, 0), 50)
            elif button_state == button_state_name[2]:
                img = cv2ImgAddText(image, f'{button_name[1]}：合闸', 10, 65, (255, 0, 0), 50)

        if line[1][0] == button_name[2]:
            if button_state == button_state_name[0]:
                img = cv2ImgAddText(image, f'{button_name[2]}：分闸', 10, 65, (255, 0, 0), 50)
            elif button_state == button_state_name[1]:
                img = cv2ImgAddText(image, f'{button_name[2]}：正常', 10, 65, (255, 0, 0), 50)
            elif button_state == button_state_name[2]:
                img = cv2ImgAddText(image, f'{button_name[2]}：合闸', 10, 65, (255, 0, 0), 50)

        if line[1][0] == button_name[3]:
            if button_state == button_state_name[0]:
                img = cv2ImgAddText(image, f'{button_name[3]}：就地', 10, 65, (255, 0, 0), 50)
            elif button_state == button_state_name[1]:
                pass
            elif button_state == button_state_name[2]:
                img = cv2ImgAddText(image, f'{button_name[3]}：远方', 10, 65, (255, 0, 0), 50)



cv2.imwrite('result1.jpg', img)
# cv2.waitKey()


# result = ocr_result[0]
# image = Image.open(image_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='doc/fonts/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('result5.jpg')






