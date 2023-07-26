from PyQt5.QtWidgets import QMainWindow, QFileDialog, QTableWidgetItem, QComboBox
from PyQt5.QtGui import QPixmap,QImage
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot
import cv2
from detection import Detection


class detect_window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setFixedSize(1280, 950) 
        loadUi('UI/wbc_final.ui',self)

        self.browseBtn.clicked.connect(self.browse_image)

        self.chooseTypeCombo.currentIndexChanged.connect(self.on_choose_type_changed)
        self.chooseTypeCombo.setCurrentIndex(0)
        self.on_choose_type_changed(self.chooseTypeCombo.currentIndex())
        
        self.radioBtnFaster.clicked.connect(self.radio_button_clicked)
        self.radioBtnYolov8.clicked.connect(self.radio_button_clicked)
        self.radioBtnDeTr.clicked.connect(self.radio_button_clicked)

        self.detection_thread = None

        if self.detection_thread is not None:
            self.detection_thread.classesCountSignal.connect(self.update_statistical_table)

    def browse_image(self):
        fname = QFileDialog.getOpenFileName(self,'Open file')
        print(fname[0])
        if fname[0]:
            self.pathLabel.setText(fname[0])
            img = cv2.imread(fname[0])
            img = cv2.resize(img,(640,640))
            qimage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qimage)
            self.img_upload.setPixmap(pixmap)


            self.img_process_label.setText("Please choose model to detect")
        else:
            self.img_process_label.setText("No image available for processing!!!")

    def on_choose_type_changed(self,index):
        selected_item = self.chooseTypeCombo.itemText(index)
        print("Selected Item: ",selected_item)


    @pyqtSlot(QImage)
    def setImg(self, image):
        self.img_upload.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(dict)
    def update_statistical_table(self,classes_count):
        #delete old data
        self.statisticalTable.clearContents()
        
        self.statisticalTable.setRowCount(len(classes_count))

        row = 0
        for cls,count in classes_count.items():
            count = classes_count[cls]
            self.statisticalTable.setItem(row, 0, QTableWidgetItem(cls))
            self.statisticalTable.setItem(row, 1, QTableWidgetItem(str(count)))
            row+=1
    def radio_button_clicked(self):
        
        if not self.check_image_selected():
            self.radioBtnFaster.setChecked(False)
            self.radioBtnYolov8.setChecked(False)
            self.radioBtnDeTr.setChecked(False)
            return

        img_path = self.pathLabel.text()
        if self.chooseTypeCombo.currentIndex() == 0:
            if self.radioBtnFaster.isChecked():
                print("Faster RCNN")
                self.img_process_label.hide()
            elif self.radioBtnYolov8.isChecked():

                print("Yolov8")
                self.img_process_label.hide()
                self.statusProcessLabel.setText("Detecting!!!")
                if self.detection_thread is not None:
                    self.detection_thread.quit()
                    self.detection_thread.wait()

                
                self.detection_thread = Detection(img_path)
                self.detection_thread.finished.connect(self.detect_completed)
                self.detection_thread.changePixmap.connect(self.setImg)
                self.detection_thread.classesCountSignal.connect(self.update_statistical_table)
                self.detection_thread.start()

            elif self.radioBtnDeTr.isChecked():
                print("DeTr")
                self.img_process_label.hide()

        if self.chooseTypeCombo.currentIndex()  == 1:
            if self.radioBtnFaster.isChecked():
                print("Faster RCNN")
                self.img_process_label.hide()
                
            elif self.radioBtnYolov8.isChecked():

                print("Yolov8")
                self.img_process_label.hide()

            elif self.radioBtnDeTr.isChecked():
                print("DeTr")
                self.img_process_label.hide()

    def check_image_selected(self):
        img_path = self.pathLabel.text()
        if not img_path:
            self.show_message_box("Please select the image before choosing the model.")
            return False
        return True

    def show_message_box(self, message):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setIcon(QtWidgets.QMessageBox.Warning)
        msgBox.setText(message)
        msgBox.setWindowTitle("Warning")
        msgBox.exec_()

    def detect_completed(self):
        return self.statusProcessLabel.setText("Completed!!!")