import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF untuk ekstraksi teks PDF


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('gui.ui', self)  # Pastikan file 'gui.ui' ada di direktori yang benar
        self.Image = None
        self.predicted_species = None  # Variabel untuk menyimpan nama spesies ikan yang diprediksi
        self.document_texts = {}  # Dictionary untuk menyimpan teks dokumen relevan

        # Hubungkan tombol dengan fungsinya
        self.uploadButton.clicked.connect(self.loadImage)
        self.identifyButton.clicked.connect(self.identifyImage)

        # Load model Keras (pastikan model dan class_names sesuai)
        try:
            self.model = load_model('FishModelClassifier.h5')
            self.class_names = [
                'Milk fish', 'Threadfin Bream', 'Giant Croaker',
                'White-finned Croaker', 'Mozambique Tilapia',
                'Nile Tilapia', 'Skipjack Tuna', 'Redspotted Croaker'
            ]
            print("Model successfully loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.class_names = []

        # Add default text to identifyLabel
        self.identifyLabel.setText("Fish Species: None")

    def loadImage(self):
        # Select image file
        root = Tk()
        root.withdraw()
        file_path = askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.tif")]
        )

        if file_path:
            self.Image = cv2.imread(file_path)
            if self.Image is not None:
                print(f"Image successfully loaded from: {file_path}")
                self.displayImage(windows=1)  # Display the image on imageLabel
            else:
                print("Failed to load image.")
        else:
            print("No file selected.")

    def displayImage(self, windows=1):
        if self.Image is not None:
            qformat = QImage.Format_Indexed8
            if len(self.Image.shape) == 3:
                if self.Image.shape[2] == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888

            img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)
            img = img.rgbSwapped()

            # Adjust the image size to match the label size
            label_width = self.imageLabel.width()
            label_height = self.imageLabel.height()
            img = img.scaled(label_width, label_height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

            if windows == 1:
                if self.imageLabel:
                    self.imageLabel.setPixmap(QPixmap.fromImage(img))
                    self.imageLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                    self.imageLabel.setScaledContents(False)
                    self.identifyLabel.setText("Fish Species:")

    def identifyImage(self):
        if self.Image is not None and self.model is not None:
            print("Identifying image...")

            # Resize the image according to the model's input size
            img_resized = cv2.resize(self.Image, (128, 128))

            # Convert the image to an array and normalize
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array / 255.0  # Normalize (if required by the model)

            # Predict the class of the image
            try:
                predictions = self.model.predict(img_array)
                predicted_index = np.argmax(predictions, axis=1)[0]

                # Display the predicted result in the identifyLabel
                self.predicted_species = self.class_names[predicted_index]
                self.identifyLabel.setText(f"Fish Species: {self.predicted_species}")

                # Call function to describe the image
                self.describeImage(self.predicted_species)

            except Exception as e:
                print(f"Error during prediction: {e}")
        else:
            print("Image not loaded or model not available for identification.")

    def describeImage(self, keyword):
        # Split the keyword into separate words
        keywords = keyword.lower().split()

        # Load documents from folder
        folder_path = 'docs'  # Path to the folder containing PDF files
        document_texts = self.extract_all_pdf_text(folder_path)
        relevant_docs = self.find_relevant_documents(document_texts, keywords)

        # Clear the List Widget before displaying new documents
        self.listWidget.clear()

        if relevant_docs:
            print(f"Relevant documents for the keyword '{keyword}':")
            for doc_title, doc_text in relevant_docs.items():
                print(f"- {doc_title}")
                # Add document title to List Widget
                self.listWidget.addItem(doc_title)

            # Save document texts in an attribute to access when selected
            self.document_texts = relevant_docs

            # Add event to handle document selection
            self.listWidget.itemClicked.connect(self.displayDocumentContent)
        else:
            print(f"No documents matched the keyword '{keyword}'.")
            self.textBrowser.setText("No matching document found.")

    def extract_all_pdf_text(self, folder_path):
        documents = {}
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                filepath = os.path.join(folder_path, filename)
                doc = fitz.open(filepath)
                text = ""
                for page in doc:
                    text += page.get_text()
                documents[filename] = text
        return documents

    def find_relevant_documents(self, document_texts, keywords):
        relevant_docs = {}

        # Iterate over all documents
        for doc_title, doc_text in document_texts.items():
            # Check if all keywords are present in the title (case insensitive)
            if all(keyword.lower() in doc_title.lower() for keyword in keywords):
                relevant_docs[doc_title] = doc_text

        return relevant_docs

    def displayDocumentContent(self, item):
        # Display the document text in Text Browser based on the selected item
        doc_title = item.text()
        if doc_title in self.document_texts:
            self.textBrowser.setText(self.document_texts[doc_title])
        else:
            self.textBrowser.setText("No document content found.")


# Run the application
app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Fish Classification')
window.show()
sys.exit(app.exec_())
