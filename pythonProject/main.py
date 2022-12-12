import math
import os
import sys

import cv2
import face_recognition
import numpy as np
from colorama import *





from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import *


class Ui_Widget(object):
    def setupUi(self, Widget):
        Widget.setObjectName("Widget")
        Widget.resize(1080, 640)

        self.image_folder = QtWidgets.QPushButton(Widget)
        self.image_folder.setGeometry(QtCore.QRect(10, 350, 200, 30))
        self.image_folder.setObjectName("image_folder")


        self.simulation = QtWidgets.QPushButton(Widget)
        self.simulation.setGeometry(QtCore.QRect(10, 400, 200, 30))
        self.simulation.setObjectName("simulation")

        self.label_video = QLabel(Widget)
        self.label_video.setGeometry(QtCore.QRect(290, 20, 311, 201))
        self.label_video.setObjectName("label_video")

        self.labelImage = QLabel(Widget)
        self.labelImage.setGeometry(QtCore.QRect(590, 20, 311, 201))
        self.labelImage.setObjectName("labelImage")

        self.retranslateUi(Widget)
        QtCore.QMetaObject.connectSlotsByName(Widget)

    def retranslateUi(self, Widget):
        _translate = QtCore.QCoreApplication.translate
        Widget.setWindowTitle(_translate("Widget", "Widget"))
        self.image_folder.setText(_translate("Widget", "Открыть фотографии сотрудников"))
        self.simulation.setText(_translate("Widget", "Включить камеру"))



class ThreadOpenCV(QThread):
    changePixmap = pyqtSignal(QImage)

#FaceRecognition
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        super().__init__()
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f"faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FPS, 24)

        while True:
            ret, frame = cap.read()
            if ret:
                if self.process_current_frame:
                    # Resize frame of video to 1/4 size for faster face recognition processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    rgb_small_frame = small_frame[:, :, ::-1]

                    # Find all the faces and face encodings in the current frame of video
                    self.face_locations = face_recognition.face_locations(rgb_small_frame)
                    self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                    self.face_names = []
                    for face_encoding in self.face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = "Unknown"
                        confidence = ''

                        # Calculate the shortest distance to face
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = face_confidence(face_distances[best_match_index])

                        # self.face_names.append(f'{name} ({confidence})')
                        self.face_names.append(f'{name}')
                self.process_current_frame = not self.process_current_frame
                for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    if name == "Unknown":
                        color_scheme = (0, 0, 255)
                    else:
                        color_scheme = (0, 255, 0)
                    # Create the frame with the name
                    cv2.rectangle(frame, (left, top), (right, bottom), color_scheme, 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color_scheme, cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

                #cv2.imshow("VIDEO",frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_expanded = np.expand_dims(frame_rgb, axis=0)
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(
                    rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(311, 201, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

                if cv2.waitKey(1) == ord('q'):
                    break

            self.msleep(20)


# +++ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


class Window(QtWidgets.QWidget, Ui_Widget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.simulation.clicked.connect(self.can)

        self.image_folder.clicked.connect(self.open)
        # +++
        self.thread = ThreadOpenCV()  # +++
        self.thread.changePixmap.connect(self.setImage)  # +++

    def can(self):
        self.thread.start()  # +++

    def setImage(self, image):  # +++
        self.label_video.setPixmap(QPixmap.fromImage(image))  # +++

    def openImage(self, image):
        pixmapImage = QPixmap(image)
        pixmapImage = pixmapImage.scaled(
            300, 300,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.labelImage.setPixmap(pixmapImage)

    def open(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            QDir.currentPath(),
            "Image Files (*.png *.jpg *.bmp)"
        )
        if fileName:
            self.image = fileName
            self.openImage(fileName)

    def imageClicked(self):
        self.window = QLabel()
        self.window.setPixmap(QPixmap(self.image))
        self.window.show()
        self.window.activateWindow()

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'



if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())


