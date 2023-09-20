# Scott Ratchford, 2023.09.19

from main import *
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QLabel, QPushButton, QFileDialog, QHBoxLayout, QVBoxLayout
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import sys
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # load the model
        try:
            if KAGGLE_MODE:
                self.loaded_model = load_model("/kaggle/working/pokemon_vs_digimon_CNN_latest.h5")
            else:
                self.loaded_model = load_model("./models/pokemon_vs_digimon_CNN_latest.h5")
        except:
            raise Exception("Could not load the model.")

        self.initUI()

    def initUI(self):
        self.setCentralWidget(Interface(self))
        self.setWindowTitle("Identifymon")
        self.resize(800, 600)
        self.showMaximized()

class Interface(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        # central widget, the image viewer
        self.image_viewer = QLabel(self)
        self.image_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # beneath the image viewer, the model's prediction
        self.prediction = QLabel(self)
        self.prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction.setText("Prediction")
        # beneath the prediction, the model's confidence
        self.confidence = QLabel(self)
        self.confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence.setText("Confidence")
        # beneath the confidence, the buttons
        self.buttons = QWidget(self)
        self.buttons.setLayout(QHBoxLayout())
        self.buttons.layout().setAlignment(Qt.AlignmentFlag.AlignCenter)
        # the buttons, from left to right
        # the left button, to select a new single image
        self.select_image_button = QPushButton(self)
        self.select_image_button.setText("Select Image")
        self.select_image_button.clicked.connect(self.select_image)
        # the right button, to select a set of images
        self.select_images_button = QPushButton(self)
        self.select_images_button.setText("Select Folder of Pokemon/Digimon Images")
        self.select_images_button.clicked.connect(self.select_images)
        # add the buttons to the buttons widget
        self.buttons.layout().addWidget(self.select_image_button)
        self.buttons.layout().addWidget(self.select_images_button)
        # add the widgets to the interface
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.image_viewer)
        self.layout().addWidget(self.prediction)
        self.layout().addWidget(self.confidence)
        self.layout().addWidget(self.buttons)

        self.show()

    def select_image(self):
        # open a file dialog to select an image
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        file_dialog.setDirectory(os.getcwd())
        if file_dialog.exec():
            # if the user selected an image, load it into the image viewer
            self.image_viewer.setPixmap(QPixmap.fromImage(QImage(file_dialog.selectedFiles()[0])))
        
        # preprocess the image
        test_image = cv2.imread(file_dialog.selectedFiles()[0])
        test_image = cv2.resize(test_image, (128, 128))
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        # make prediction
        prediction = self.parent().loaded_model.predict(test_image)
        predicted_label = np.argmax(prediction)
        confidence = prediction[0][predicted_label] * 100  # Convert to percentage

        # output the result
        if predicted_label == 0:
            self.prediction.setText("The model predicts this image is a Pokemon.")
        else:
            self.prediction.setText("The model predicts this image is a Digimon.")
        self.confidence.setText(f"Confidence: {confidence:.2f}%")

    def select_images(self):
        # open a file dialog to select a folder of images
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)
        file_dialog.setDirectory(os.getcwd())
        if file_dialog.exec():
            # if the user selected a folder, load the first image in the folder into the image viewer
            self.image_viewer.setPixmap(QPixmap.fromImage(QImage(file_dialog.selectedFiles()[0] + "/" + os.listdir(file_dialog.selectedFiles()[0])[0])))
        
        # Load and preprocess every image in the test folder
        test_images = []
        test_filenames = []
        for filename in os.listdir(file_dialog.selectedFiles()[0]):
            img = cv2.imread(os.path.join(file_dialog.selectedFiles()[0], filename))
            if img is not None:
                img = cv2.resize(img, (128, 128))
                img = img / 255.0
                test_images.append(img)
                test_filenames.append(filename)

        # Convert the list of images to a NumPy array
        test_images = np.array(test_images)
        # make predictions
        predictions = self.parent().loaded_model.predict(test_images)
        predicted_labels = np.argmax(predictions, axis=1)
        confidences = []
        for prediction in predictions:
            confidences.append(prediction[predicted_labels[0]] * 100)
        
        # output the result
        if predicted_labels[0] == 0:
            self.prediction.setText("The model predicts these images are Pokemon.")
        else:
            self.prediction.setText("The model predicts these images are Digimon.")
        self.confidence.setText(f"Average Confidence: {sum(confidences) / len(confidences):.2f}%")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
