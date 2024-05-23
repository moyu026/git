import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
from torchvision import models
from torchsummary import summary
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os


def myimshows(imgs, titles=False, fname="", size=6):
    lens = len(imgs)
    fig = plt.figure(figsize=(size * lens, size))
    if titles == False:
        titles = "0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap='Reds')
        else:
            plt.imshow(imgs[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(fname, bbox_inches='tight')
    # plt.show()


def tensor2img(tensor, heatmap=False, shape=(224, 224)):
    np_arr = tensor.detach().numpy()[0]
    # 对数据进行归一化
    if np_arr.max() > 1 or np_arr.min() < 0:
        np_arr = np_arr - np_arr.min()
        np_arr = np_arr / np_arr.max()
    np_arr = (np_arr * 255).astype(np.uint8)
    if np_arr.shape[0] == 1:
        np_arr = np.concatenate([np_arr, np_arr, np_arr], axis=0)
    np_arr = np_arr.transpose((1, 2, 0))
    if heatmap:
        np_arr = cv2.resize(np_arr, shape)
        np_arr = cv2.applyColorMap(np_arr, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    return np_arr / 255


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())
    # print("backward_hook:", grad_in[0].shape, grad_out[0].shape)


def farward_hook(module, input, output):
    fmap_block.append(output)
    # print("farward_hook:", input[0].shape, output.shape)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model_button_path = r'F:\PythonWork\5.classification\1.resnet\weight\result_model\resNet18_button3.pth'

class_names_button = ['left', 'left_top', 'right_top', 'top']
num_classes = len(class_names_button)
net_shell = torchvision.models.resnet18(False).eval()
layerFC = net_shell.fc.in_features  # 修改最后一层的分类输出层
net_shell.fc = torch.nn.Linear(layerFC, num_classes)  # 512 --> 2
net_shell.load_state_dict(torch.load(model_button_path, map_location=torch.device("cpu")))
model = net_shell

# model = models.resnet18(pretrained=True)
# model.eval()  # 评估模式
# summary(model,input_size=(3,512,512))

# 注册hook
fh = model.layer4.register_forward_hook(farward_hook)
bh = model.layer4.register_backward_hook(backward_hook)

# 定义存储特征和梯度的数组
fmap_block = list()
grad_block = list()

# 加载变量并进行预测
img_folder = r"F:\PythonWork\dataset\electrical_cabinet_all\classify\button\button_img\top"
save_img_folder = r'F:\PythonWork\dataset\electrical_cabinet_all\classify\button\button_classify_result\Visualization\resNet18_button3\top'

img_list = os.listdir(img_folder)

for imgs in img_list:
    img_path = os.path.join(img_folder, imgs)
    save_img_path = os.path.join(save_img_folder, imgs)
    img = cv2.imread(img_path)

    print(img.shape)
    img_shape = (img.shape[1], img.shape[0])

    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    image_tensor = transform(img).unsqueeze(0)  # 添加批量维度
    image_tensor = image_tensor

    preds = model(image_tensor)

    # 构造label，并进行反向传播
    clas = 3  #类别个数
    trues = torch.ones((1,), dtype=torch.int64) * clas
    ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(preds, trues)
    loss.backward()

    # 卸载hook
    fh.remove()
    bh.remove()

    # 取出相应的特征和梯度
    layer1_grad = grad_block[-1]  # layer1_grad.shape [1, 64, 128, 128]
    layer1_fmap = fmap_block[-1]

    # 将梯度与fmap相乘
    cam = layer1_grad[0, 0].mul(layer1_fmap[0, 0])
    for i in range(1, layer1_grad.shape[1]):
        cam += layer1_grad[0, i].mul(layer1_fmap[0, i])
    layer1_grad = layer1_grad.sum(1, keepdim=True)  # layer1_grad.shape [1, 1, 128, 128]
    layer1_fmap = layer1_fmap.sum(1, keepdim=True)  # 为了统一在tensor2img函数中调用
    cam = cam.reshape((1, 1, *cam.shape))

    # 进行可视化
    img_np = tensor2img(image_tensor)
    # layer1_fmap=torchvision.transforms.functional.resize(layer1_fmap,[224, 224])
    layer1_grad_np = tensor2img(layer1_grad, heatmap=True, shape=img_shape)
    layer1_fmap_np = tensor2img(layer1_fmap, heatmap=True, shape=img_shape)
    cam_np = tensor2img(cam, heatmap=True, shape=img_shape)

    myimshows([img_np, cam_np, cam_np * 0.4 + img_np * 0.6], ['image', 'cam', 'cam + image'], fname=save_img_path)
