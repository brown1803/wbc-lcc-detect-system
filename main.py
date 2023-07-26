from PyQt5.QtWidgets import QApplication
import sys
from detect_window import detect_window


app = QApplication(sys.argv)

window = detect_window()
window.show()

try:
    sys.exit(app.exec_()) 
except:
    print("Exiting")