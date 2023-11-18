from ultralytics import YOLO 
from PyQt5.QtCore import QThread,pyqtSignal,Qt 
from PyQt5.QtGui import QImage
import cv2
import numpy as np

classes = ['basophil', 'eosinophil', 'lymphocyte', 'monocyte', 'neutrophil','rbc']
classes_to_detect = ['basophil', 'eosinophil', 'lymphocyte', 'monocyte', 'neutrophil','rbc']
class BloodYolo(QThread):
    def __init__(self,imgPath):
        super(BloodYolo, self).__init__()
        self.yolo = YOLO("model_file/yolo/blood.pt")
        self.imgPath = imgPath

    changePixmap = pyqtSignal(QImage)
    classesCountSignal = pyqtSignal(dict)

    def run(self):
        img = cv2.imread(self.imgPath)
        height, width, channel = img.shape
        results = self.yolo(img)

        classes_count = {cls: 0 for cls in classes}

        for result in results:
            boxes = result.boxes.numpy()
            for box in boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
               
                print("Object: ",classes[cls]," with ",conf," confident score")

                if conf > 0.0:
                    if classes[cls] in classes:
                        x, y, x2, y2 = map(int,box.xyxy[0])
                        # print("Cls: ",classes[cls])
                        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f"{classes[cls]}", (x+3, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        
                        
                        classes_count[classes[cls]] += 1

                        print("Object: ",classes[cls]," with ",conf," confident score")
        
        print("Quantity each class: ")

        for cls, count in classes_count.items():
            print(f"{cls} : {count}")

        

        rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bytesPerLine = channel * width
        convertToQtFormat = QImage(rgbImage.data, width, height, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(600, 600, Qt.KeepAspectRatio)
        self.changePixmap.emit(p)
        self.classesCountSignal.emit(classes_count)
        