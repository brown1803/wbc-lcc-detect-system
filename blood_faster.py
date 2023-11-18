from ultralytics import YOLO 
from PyQt5.QtCore import QThread,pyqtSignal,Qt 
from PyQt5.QtGui import QImage
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def convert_to_lowercase(text):
    lowercase_text = ""
    for char in text:
        if char == 'G':
            lowercase_text += 'g'
        else:
            lowercase_text += char.lower()
    return lowercase_text

def label_text2int(t):
    low = convert_to_lowercase(t)
    if( low == 'eosinophil' or low == 'eosino' ): return 1 # tiểu cầu
    elif( low == 'lymphocyte' or low == 'lympho'): return 2
    elif( low == 'monocyte' or low == 'mono' ): return 3
    elif( low == 'neutrophil' or low == 'neutro' ): return 4
    elif( low == 'basophil' or low == 'baso' ): return 5
    elif( low == 'rbc' ): return 6
    elif( low == 'platelet' or low == 'platelets' ): return 7
    else: return 0
    return 0

def label_int2text(i):
    if( i == 1 ): return 'EOSINOPHIL'
    elif( i == 2 ): return 'LYMPHOCYTE'
    elif( i == 3 ): return 'MONOCYTE'
    elif( i == 4 ): return 'NEUTROPHIL'
    elif( i == 5 ): return 'BASOPHIL'
    elif( i == 6 ): return 'RBC'
    elif( i == 7 ): return 'PLATELET'
    else: return 'NAN'
    return 'NAN'

classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL', 'BASOPHIL','RBC']
num_class = 8
imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_class)
state_dict = torch.load('model_file/faster/blood.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

class BloodFaster(QThread):
    def __init__(self,imgPath):
        super(BloodFaster, self).__init__()
        self.imgPath = imgPath

    changePixmap = pyqtSignal(QImage)
    classesCountSignal = pyqtSignal(dict)
        

    def run(self):
        classes_count = {cls: 0 for cls in classes}
        model.eval()

        img = cv2.imread(self.imgPath)
        # img = cv2.resize(img, (840, 840))
        height, width, channel = img.shape
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Convert the image to an array
        image = img_to_array(image)
        resized_image_array = image.astype(np.float32) / 255.0
        nor = (resized_image_array - imagenet_mean) / imagenet_std

        X = []
        X.append(nor)
        X = np.array(X)
        X = torch.tensor(X, dtype=torch.float32)
        X = X.permute(0, 3, 1, 2)

        results = model(X)

        box = results[0]['boxes']
        clas = results[0]['labels']
        scor = results[0]['scores']

        for i in range(len(box)):
            if scor[i] > 0.5:
                x = int(box[i][0])
                y = int(box[i][1])
                x2 = int(box[i][2])
                y2 = int(box[i][3])
                label1 = label_int2text(clas[i])
                label2 = str(round(scor[i].item(), 2))
                cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label1, (x+3, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                print("Object: ", label1," with ", scor[i]," confident score")
                classes_count[label1] += 1

        rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bytesPerLine = channel * width
        convertToQtFormat = QImage(rgbImage.data, width, height, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(600, 600, Qt.KeepAspectRatio)
        self.changePixmap.emit(p)
        self.classesCountSignal.emit(classes_count)
        