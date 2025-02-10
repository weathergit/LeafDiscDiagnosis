import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
from torchvision.transforms import v2
from ultralytics import YOLO, SAM
import segmentation_models_pytorch as seg

device = "cuda" if torch.cuda.is_available() else "cpu"


class LeafDownSegPipe:
    def __init__(self, img_path):
        self.imgpath = img_path
        self.yolo_path = './weights/yolon_best.pt'
        self.mobile_sam_path = './weights/mobile_sam.pt'
        self.unet_path = './weights/UNet_best.pth'

        self.img = cv2.imread(self.imgpath)
        self.basename = os.path.basename(self.imgpath)[:-4]

    def get_yolo_output(self):
        yolo = YOLO(self.yolo_path)
        yolo_results = yolo.predict(self.img, conf=0.25, iou=0.6, show=False, imgsz=640)
        for result in yolo_results:
            boxes = result.boxes
            bbox = boxes.xyxy.cpu().numpy()
        return bbox

    def get_sam_output(self):
        mobile_sam = SAM(self.mobile_sam_path)
        bbox_arr = self.get_yolo_output()
        pts_list = []
        clip_images = {}
        yolo_res = self.img.copy()
        for i, bbox in enumerate(bbox_arr):
            templates = self.img.copy()
            leaf_disc = mobile_sam.predict(source=templates, bboxes=bbox)
            pts = leaf_disc[0].masks.xy[0]
            pts_list.append(pts)
            rect = cv2.boundingRect(pts)
            x, y, w, h = rect
            croped = templates[y:y + h, x:x + w]
            cv2.imwrite('{0}.jpg'.format(i), croped)
            yolo_res = cv2.rectangle(np.int32(yolo_res), (x, y), (x+w, y+h),color=(57, 0, 199) , thickness=2)
            # yolo_res = cv2.polylines(np.int32(yolo_res),pts=[pts.astype('int32')],isClosed=True,color=(0, 191, 255), thickness=2)
            pts1 = pts - pts.min(axis=0)
            pts1 = pts1.astype('int32')
            mask = np.zeros(croped.shape[:2], np.uint8)
            cv2.drawContours(mask, [pts1], -1, (255, 255, 255), -1, cv2.LINE_AA)
            dst = cv2.bitwise_and(croped, croped, mask=mask)
            clip_images[rect] = dst
            # cv2.imwrite('{0}_clip.jpg'.format(i), dst.astype('uint8'))
        # cv2.imwrite('yolo_bbox.jpg', yolo_res.astype('uint8'))
        return pts_list, clip_images

    def get_unet_output(self):
        encoder = 'resnet34'
        encoder_weight = 'imagenet'
        preprocessing = seg.encoders.get_preprocessing_fn(encoder_name=encoder, pretrained=encoder_weight)
        UNet = torch.load(self.unet_path, map_location=device)
        UNet.eval()
        _, clip_images = self.get_sam_output()
        masks_list = []
        for key, img in clip_images.items():
            img = img[:, :, ::-1]
            height, width, _ = img.shape
            img_resize = cv2.resize(img, (224, 224))
            img_pre = preprocessing(img_resize)
            img_pre = img_pre.transpose(2, 0, 1).astype('float32')
            img_tensor = torch.from_numpy(img_pre)
            img_tensor = img_tensor.unsqueeze(0)
            masks = UNet.predict(img_tensor.to(device))
            masks_arr = torch.round(masks.squeeze(0))
            masks_list.append(masks_arr)
        return masks_list

    def get_results(self, severity=True):
        pts_list, clip_images = self.get_sam_output()
        mask_list = self.get_unet_output()
        out_image = self.img.copy()
        out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
        for i, (items, mask) in enumerate(zip(clip_images.items(), mask_list)):
            rect, img = items
            x, y, w, h = rect
            img = torch.from_numpy(img.transpose(2, 0, 1))
            _, height, width = img.shape
            img = v2.Resize(size=(224, 224), antialias=True)(img)
            mask = mask.bool()
            with_masks = draw_segmentation_masks(img, masks=mask, alpha=0.3, colors=['green', 'red'])
            out_img = F.to_pil_image(with_masks)
            out_img = out_img.resize((width, height))
            # out_img.save('{0}_mask.jpg'.format(i))
            out_image[y:y+h, x:x+w] = np.asarray(out_img)
    

        temp_mask = np.zeros(out_image.shape[:2], dtype=np.uint8)
        for pts in pts_list:
            pts = pts.astype('int32')
            cv2.drawContours(temp_mask, [pts], -1, (255, 255, 255), -1)
            # cv2.drawContours(out_image, [pts], -1, (255, 255, 255), -1)

        cv2.bitwise_not(out_image, out_image)
        dst = cv2.bitwise_not(out_image, self.img, mask=temp_mask)[..., ::-1]
        cv2.imwrite(self.basename + '_result.jpg', dst)
        if severity:
            for rect, mask in zip(clip_images.keys(), mask_list):
                x, y, w, h = rect
                mask = mask.bool().cpu().numpy().astype(int)
                downy = mask[1, :, :].sum()
                leaf = mask[0, :, :].sum()
                severity_ration = downy / (downy + leaf) * 100
                dst = cv2.rectangle(np.int32(dst), (x, y), (x+w, y+h), (221, 28, 119), 1)
                cv2.putText(np.int32(dst), 'Severity:{:.2f}'.format(severity_ration), (x, y-2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (221, 28, 119), 1)
            dst = dst.astype('uint8')
            cv2.imwrite(self.basename + '_result_box.jpg', dst)
            cv2.namedWindow('results_with_box', cv2.WINDOW_NORMAL)
            cv2.imshow("results_with_box", dst)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path = './16-1-1-1(27).jpg'
    model = LeafDownSegPipe(img_path=img_path)
    # yolo_out = model.get_yolo_output()
    # print(yolo_out)
    # sam_output = model.get_sam_output()
    # print(sam_output)
    # model.get_results()
    # print(len(sam_output))
    # print(sam_output.shape)
