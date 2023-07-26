from ultralytics import YOLO 
from PyQt5.QtCore import QThread,pyqtSignal,Qt 
from PyQt5.QtGui import QImage
import cv2
import numpy as np

classes = ['basophil', 'eosinophil', 'lymphocyte', 'monocyte', 'neutrophil','platelet', 'platelets', 'rbc']
classes_to_detect = ['basophil', 'eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']
class Detection(QThread):
    def __init__(self,imgPath):
        super(Detection, self).__init__()
        self.yolo = YOLO("model_file/model.pt")
        self.imgPath = imgPath

    changePixmap = pyqtSignal(QImage)
    classesCountSignal = pyqtSignal(dict)

    def run(self):
        img = cv2.imread(self.imgPath)
        height, width, channel = img.shape
        results = self.yolo(img)

        classes_count = {cls: 0 for cls in classes_to_detect}

        for result in results:
            boxes = result.boxes.numpy()
            for box in boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
               
                if classes[cls] in classes_to_detect:
                    x, y, x2, y2 = map(int,box.xyxy[0])
                    cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{classes[cls]}", (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                    
                    classes_count[classes_to_detect[cls]] += 1

                    print("Object: ",classes[cls]," with ",conf," confident score")

        print("Quantity each class: ")

        for cls, count in classes_count.items():
            print(f"{cls} : {count}")

        

        rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bytesPerLine = channel * width
        convertToQtFormat = QImage(rgbImage.data, width, height, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(854, 480, Qt.KeepAspectRatio)
        self.changePixmap.emit(p)
        self.classesCountSignal.emit(classes_count)
        