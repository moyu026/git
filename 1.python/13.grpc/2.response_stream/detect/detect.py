import torch
import torchvision
from util.plot_img import draw_img, judge_detect_name
from torchvision import transforms
import numpy as np

from pointer.util.augmentation import BaseTransform
from pointer.util.config import config as cfg, update_config
from pointer.util.option import BaseOptions
from pointer.network.textnet import TextNet
from pointer.util.detection_mask import TextDetector as TextDetector_mask
from pointer.util.misc import to_device
from pointer.util.read_meter import MeterReader
from pointer.util.converter import keys, StringLabelConverter


class Classsification:
    def __init__(self, model_button_path, model_yaban_path):
        self.model_button_path = model_button_path
        self.model_yaban_path = model_yaban_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names_button = ['左', '左上', '右上', '上']
        self.class_names_yaban = ['关', '开']
        self.button_result = {}
        self.yaban_result = {}

    def button_classify(self, img, button_dict):
        num_button_classes = len(self.class_names_button)
        net_button = torchvision.models.resnet18(False).eval()
        layerFC = net_button.fc.in_features
        net_button.fc = torch.nn.Linear(layerFC, num_button_classes)
        net_button.load_state_dict(torch.load(self.model_button_path))
        model_button = net_button.to(self.device)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for button_name in button_dict:
            button_point = button_dict[button_name]
            xmin, ymin, xmax, ymax = int(button_point[0]), int(button_point[1]), int(button_point[2]), int(button_point[3])
            button_img = img[ymin:ymax, xmin:xmax]

            button_image_tensor = transform(button_img).unsqueeze(0)
            button_image_tensor = button_image_tensor.to(self.device)

            results = model_button(button_image_tensor)
            _, predicted_idx = torch.max(results, 1)
            classify_result = self.class_names_button[predicted_idx]

            left_top = (xmin, ymin)
            right_bottom = (xmax, ymax)
            img = draw_img(img, classify_result, left_top, right_bottom)

            self.button_result[button_name] = classify_result

        return self.button_result, img

    def yaban_classify(self, img, yaban_dict):
        num_yaban_classes = len(self.class_names_yaban)
        net_yaban = torchvision.models.resnet50(False).eval()
        layerFC = net_yaban.fc.in_features
        net_yaban.fc = torch.nn.Linear(layerFC, num_yaban_classes)
        net_yaban.load_state_dict(torch.load(self.model_yaban_path))
        model_yaban = net_yaban.to(self.device)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for yaban_name in yaban_dict:
            yaban_point = yaban_dict[yaban_name]
            xmin, ymin, xmax, ymax = int(yaban_point[0]), int(yaban_point[1]), int(yaban_point[2]), int(yaban_point[3])
            yaban_img = img[ymin:ymax, xmin:xmax]

            yaban_image_tensor = transform(yaban_img).unsqueeze(0)
            yaban_image_tensor = yaban_image_tensor.to(self.device)

            results = model_yaban(yaban_image_tensor)
            _, predicted_idx = torch.max(results, 1)
            classify_result = self.class_names_yaban[predicted_idx]

            left_top = (xmin, ymin)
            right_bottom = (xmax, ymax)
            img = draw_img(img, classify_result, left_top, right_bottom)
            self.yaban_result[yaban_name] = classify_result

        return self.yaban_result, img


class Detection:
    def __init__(self, model_small_lightpath):
        self.model_small_light_path = model_small_lightpath
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.small_light_result = {}
        self.light_result = {}

    def small_light_detect(self, img, small_light_dict, monitor_dict, protection_dict, binglie_dict):
        model_small_light = torch.hub.load(r'./yolov5/', 'custom', source='local', path=self.model_small_light_path)
        model_small_light.to(self.device)

        if len(monitor_dict) != 0:
            for monitor_name in monitor_dict:
                monitor_point = monitor_dict[monitor_name]
                xmin, ymin, xmax, ymax = int(monitor_point[0]), int(monitor_point[1]), int(monitor_point[2]), int(monitor_point[3])
                monitor_img = img[ymin:ymax, xmin:xmax]

                results = model_small_light(monitor_img)
                detections = results.xyxy[0]
                for detection in detections:
                    det_xmin, det_ymin, det_xmax, det_ymax, confidence, class_id = detection.tolist()
                    det_xmin, det_ymin, det_xmax, det_ymax = int(det_xmin), int(det_ymin), int(det_xmax), int(det_ymax)
                    left_top = (det_xmin + xmin, det_ymin + ymin)
                    right_bottom = (det_xmax + xmin, det_ymax + ymin)
                    if class_id == 1:  # 非绿灯
                        label = '红'
                        img = draw_img(img, label, left_top, right_bottom)
                        small_light_name = judge_detect_name(left_top, right_bottom, small_light_dict)
                        self.small_light_result[small_light_name] = label
                    if class_id == 0:  # 绿灯
                        label = '绿'
                        img = draw_img(img, label, left_top, right_bottom)
                        small_light_name = judge_detect_name(left_top, right_bottom, small_light_dict)
                        self.small_light_result[small_light_name] = label

        if len(protection_dict) != 0:
            for protection_name in protection_dict:
                protection_point = protection_dict[protection_name]
                xmin, ymin, xmax, ymax = int(protection_point[0]), int(protection_point[1]), int(protection_point[2]), int(protection_point[3])
                protection_img = img[ymin:ymax, xmin:xmax]

                results = model_small_light(protection_img)
                detections = results.xyxy[0]
                for detection in detections:
                    det_xmin, det_ymin, det_xmax, det_ymax, confidence, class_id = detection.tolist()
                    det_xmin, det_ymin, det_xmax, det_ymax = int(det_xmin), int(det_ymin), int(det_xmax), int(det_ymax)
                    left_top = (det_xmin + xmin, det_ymin + ymin)
                    right_bottom = (det_xmax + xmin, det_ymax + ymin)
                    if class_id == 1:  # 非绿灯
                        label = '红'
                        img = draw_img(img, label, left_top, right_bottom)
                        small_light_name = judge_detect_name(left_top, right_bottom, small_light_dict)
                        self.small_light_result[small_light_name] = label
                    if class_id == 0:  # 绿灯
                        label = '绿'
                        img = draw_img(img, label, left_top, right_bottom)
                        small_light_name = judge_detect_name(left_top, right_bottom, small_light_dict)
                        self.small_light_result[small_light_name] = label

        if len(binglie_dict) != 0:
            for binglie_name in binglie_dict:
                binglie_point = binglie_dict[binglie_name]
                xmin, ymin, xmax, ymax = int(binglie_point[0]), int(binglie_point[1]), int(binglie_point[2]), int(binglie_point[3])
                binglie_img = img[ymin:ymax, xmin:xmax]

                results = model_small_light(binglie_img)
                detections = results.xyxy[0]
                for detection in detections:
                    det_xmin, det_ymin, det_xmax, det_ymax, confidence, class_id = detection.tolist()
                    det_xmin, det_ymin, det_xmax, det_ymax = int(det_xmin), int(det_ymin), int(det_xmax), int(det_ymax)
                    left_top = (det_xmin + xmin, det_ymin + ymin)
                    right_bottom = (det_xmax + xmin, det_ymax + ymin)
                    if class_id == 1:  # 非绿灯
                        label = '红'
                        img = draw_img(img, label, left_top, right_bottom)
                        small_light_name = judge_detect_name(left_top, right_bottom, small_light_dict)
                        self.small_light_result[small_light_name] = label
                    if class_id == 0:  # 绿灯
                        label = '绿'
                        img = draw_img(img, label, left_top, right_bottom)
                        small_light_name = judge_detect_name(left_top, right_bottom, small_light_dict)
                        self.small_light_result[small_light_name] = label

        return self.small_light_result, img

    # def light_detect(self, img, light_dict):
    #     model_light = torch.hub.load(r'./yolov5/', 'custom', source='local', path=self.model_light_path)
    #     model_light.to(self.device)
    #
    #     results = model_light(img)
    #     detections = results.xyxy[0]
    #     for detection in detections:
    #         xmin, ymin, xmax, ymax, confidence, class_id = detection.tolist()
    #         xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    #         left_top = (xmin, ymin)
    #         right_bottom = (xmax, ymax)
    #         if class_id == 1:  # 绿灯
    #             label = 'green'
    #             img = draw_img(img, label, left_top, right_bottom)
    #             light_name = judge_detect_name(left_top, right_bottom, light_dict)
    #             self.light_result[light_name] = label
    #         if class_id == 0:  # 红灯
    #             label = 'red'
    #             img = draw_img(img, label, left_top, right_bottom)
    #             light_name = judge_detect_name(left_top, right_bottom, light_dict)
    #             self.light_result[light_name] = label
    #
    #     return self.light_result, img


class Pointer:
    def __init__(self, model_pointer_path):
        self.model_pointer_path = model_pointer_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pointer_result = {}

    def pointer_detect(self, img, pointer_dict):

        option = BaseOptions()
        args = option.initialize()

        update_config(cfg, args)

        model_path = self.model_pointer_path

        model = TextNet(is_training=False, backbone=cfg.net)

        model.load_model(model_path)
        model = model.to(cfg.device)
        converter = StringLabelConverter(keys)

        # det = Detector()
        detector = TextDetector_mask(model)
        meter = MeterReader()
        transform = BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)

        for pointer_name in pointer_dict:
            pointer_point = pointer_dict[pointer_name]
            xmin, ymin, xmax, ymax = int(pointer_point[0]), int(pointer_point[1]), int(pointer_point[2]), int(pointer_point[3])
            pointer_img = img[ymin:ymax, xmin:xmax]

            image, _ = transform(pointer_img)
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).unsqueeze(0)
            image = to_device(image)

            output = detector.detect1(image)

            pointer_pred, dail_pred, text_pred, preds, std_points, number_boxes = output['pointer'], output['dail'], output['text'], output['reco'], output['std'], output['boxes']

            pred, preds_size = preds
            if pred != None:
                _, pred = pred.max(2)
                pred = pred.transpose(1, 0).contiguous().view(-1)
                pred_transcripts = converter.decode(pred.data, preds_size.data, raw=False)
                pred_transcripts = [pred_transcripts] if isinstance(pred_transcripts, str) else pred_transcripts
            else:
                pred_transcripts = None

            img_show = image[0].permute(1, 2, 0).cpu().numpy()
            img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

            pointer_number = meter(img_show, pointer_pred, dail_pred, text_pred, pred_transcripts, std_points, number_boxes)

            left_top = (xmin, ymin)
            right_bottom = (xmax, ymax)
            img = draw_img(img, pointer_number, left_top, right_bottom)

            self.pointer_result[pointer_name] = pointer_number

        return self.pointer_result, img




