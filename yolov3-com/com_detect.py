import argparse
import shutil
import time
from pathlib import Path
from sys import platform

import sys 
sys.path.append('./yolov3-com/')
from com_models import *
from com_utils.letterbox import letterbox

class Yolov3Detect: 
    def __init__(self, cfg, weights):
        self.img_size = 416
        self.cfg = cfg
        self.weights = weights
        self.model = self.load_model()
        
        
    def load_model(self):
        # Initialize model
        model = Darknet(self.cfg, self.img_size)
        
        # Load weights
        if self.weights.endswith('.pt'):  # pytorch format
            if self.weights.endswith('yolov3.pt') and not os.path.exists(self.weights):
                if (platform == 'darwin') or (platform == 'linux'):
                    os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + self.weights)
            model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, self.weights)

        return model.cuda().eval()

def createInput (img_cv2,img_size=416):
    # Padded resize
    img, _, _, _ = letterbox(img_cv2, height=img_size)

    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    return img,img_cv2
        
def detect(model,images,img_size=416,conf_thres=0.3,nms_thres=0.45,):
    with torch.no_grad():

        # Get classes and colors
        classes = load_classes('yolov3-com/data/coco.names')

        img,im0 = images

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).cuda()

        pred = model(img)
        pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes < threshold

        if len(pred) > 0:
            # Run NMS on predictions
            detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]

            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            return detections.cpu().data.numpy()


if __name__ == '__main__':
    yolo_cfg = './cfg/yolov3.cfg'
    weightsPath = './weights/yolov3.weights'
    
    com_detect = Yolov3Detect(yolo_cfg,weightsPath)
    
    imgPath = '/home/datalab/ex_disk/work/shengdan/jupyter/pytorch/OCR-dectect/seglink/material/65.jpg'
    img_cv2 = cv2.imread(imgPath)
    yinput = createInput (img_cv2)
    res = detect(com_detect.model,yinput)
    print(res)