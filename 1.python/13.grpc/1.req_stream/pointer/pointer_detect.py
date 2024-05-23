import os
import cv2
import numpy as np
from util.augmentation import BaseTransform
from util.config import config as cfg, update_config
from util.option import BaseOptions
from network.textnet import TextNet
from util.detection_mask import TextDetector as TextDetector_mask
import torch
from util.misc import to_device
from util.read_meter import MeterReader
from util.converter import keys, StringLabelConverter


option = BaseOptions()
args = option.initialize()

update_config(cfg, args)


predict_dir = r'F:\PythonWork\1.detection\5.eletrical_cabinet\pointer\Detect-and-read-meters-2\pointer_img'
model_path = 'trian_weights/3/textgraph_vgg_80.pth'

model = TextNet(is_training=False, backbone=cfg.net)
# model_path = os.path.join(cfg.save_dir, cfg.exp_name,'textgraph_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))

model.load_model(model_path)
model = model.to(cfg.device)
converter = StringLabelConverter(keys)

# det = Detector()
detector = TextDetector_mask(model)
meter = MeterReader()
transform = BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)

image_list = os.listdir(predict_dir)

for index in image_list:
    print("**************", index)
    img_path = os.path.join(predict_dir, index)
    image = cv2.imread(img_path)

    image, _ = transform(image)  # 对检测到的指针目标图片做变换 [694,719,3]->[672,704,3]
    image = image.transpose(2, 0, 1)  # [672,704,3]->[3,672,704]
    image = torch.from_numpy(image).unsqueeze(0)
    image = to_device(image)

    output = detector.detect1(image)

    pointer_pred, dail_pred, text_pred, preds, std_points, number_boxes = output['pointer'], output['dail'], output['text'], output['reco'], output['std'], output['boxes']

    # decode predicted text
    pred, preds_size = preds
    if pred != None:
        _, pred = pred.max(2)
        pred = pred.transpose(1, 0).contiguous().view(-1)
        pred_transcripts = converter.decode(pred.data, preds_size.data, raw=False)
        pred_transcripts = [pred_transcripts] if isinstance(pred_transcripts, str) else pred_transcripts
        # print("preds", pred_transcripts)
    else:
        pred_transcripts = None

    img_show = image[0].permute(1, 2, 0).cpu().numpy()
    img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

    # print(pred_transcripts)
    # cv2.imwrite('pointer.jpg', pointer_pred)

    result = meter(img_show, pointer_pred, dail_pred, text_pred, pred_transcripts, std_points, number_boxes)
