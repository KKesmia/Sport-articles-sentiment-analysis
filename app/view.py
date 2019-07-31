import sys
#add path of the project folder on your laptop here 
#sys.path.append('C:/Users/Moi/Desktop/AARN_2')
sys.path.append('YOUR PATH')

from app.tools import treattext
from PyQt5.QtCore import  Qt
from PyQt5.QtWidgets import (QApplication, QGridLayout,QDialog, QGroupBox, QHBoxLayout,QPushButton,QSizePolicy,QVBoxLayout, QWidget, QPlainTextEdit)
from keras.models import load_model


loaded_model = load_model('../Script/best_model.h5')

class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        self.originalPalette = QApplication.palette()
        self.createRightGroupBox()
        self.createLeftWidget()

        topLayout = QVBoxLayout()
        topLayout.addStretch(1)

        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        mainLayout.addWidget(self.RightGroupBox, 1, 1)
        mainLayout.addWidget(self.LeftWidget, 1, 0)
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 1)
        self.setLayout(mainLayout)
        self.setWindowTitle("Text Objectivity Analysis")

    def Exit(self, styleName):
        sys.exit(app.exec_()) 

    def Execute(self):
        data = treattext(self.textEdit.toPlainText())
        respond = loaded_model.predict(data)
        respond = (round(respond[0][0]) == 0  )*'Text is objective' +  (round(respond[0][0]) == 1)*'Text is subjective'
        self.textEdit1.setPlainText(respond+'\n'+self.textEdit1.toPlainText())

    def createRightGroupBox(self):
        self.RightGroupBox = QGroupBox()

        button1 = QPushButton("Execute")
        button1.clicked.connect(self.Execute)
        button2 = QPushButton("Exit")
        button2.clicked.connect(self.Exit)

        self.textEdit1 = QPlainTextEdit()
        self.textEdit1.setReadOnly(True)
        self.textEdit1.setPlainText('Model loaded.')



        layout = QVBoxLayout()

        layout.addWidget(self.textEdit1)
        layout.addWidget(button1)
        layout.addWidget(button2)
        self.RightGroupBox.setLayout(layout)


    def createLeftWidget(self):
        self.LeftWidget = QGroupBox()
        self.LeftWidget.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Ignored)
        
        self.textEdit = QPlainTextEdit()
        self.textEdit.setPlainText("Type your text here !")

        tab2hbox = QHBoxLayout()
        tab2hbox.setContentsMargins(5, 5, 5, 5)
        tab2hbox.addWidget(self.textEdit)
        self.LeftWidget.setLayout(tab2hbox)


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.show()

sys.exit(app.exec_()) 
