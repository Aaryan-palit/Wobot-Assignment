import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import shutil

# Build a CNN Model for Face Recognition
class FaceRecognitionCNN:
    def __init__(self, inputShape=(224, 224, 3)):
        self.inputShape = inputShape
        self.model = self.buildModel()

    def buildModel(self):
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.inputShape),
            layers.MaxPooling2D(2, 2),

            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),

            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Output layer for binary classification (face vs non-face)
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def preprocessImage(self, imagePath):
        image = cv2.imread(imagePath)
        if image is None:
            return None
        # Resize and normalize the image
        image = cv2.resize(image, (self.inputShape[0], self.inputShape[1]))
        return image / 255.0  # Normalize

    def trainModel(self, datasetPath):
        images = []
        labels = []
        personId = 0

        # Load the dataset and assign unique IDs for each person
        for filename in os.listdir(datasetPath):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                imagePath = os.path.join(datasetPath, filename)
                processedImage = self.preprocessImage(imagePath)

                if processedImage is not None:
                    images.append(processedImage)
                    labels.append(personId)
                    personId += 1

        if not images:
            raise ValueError("No valid faces found in the dataset")

        X = np.array(images)
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    def predictImage(self, imagePath):
        processedImage = self.preprocessImage(imagePath)
        if processedImage is None:
            return None
        prediction = self.model.predict(np.array([processedImage]))
        return int(prediction[0][0])

# Main class to handle the face classification system
class FaceClassificationSystem:
    def __init__(self, inputPath, outputPath):
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.faceRecognizer = FaceRecognitionCNN()

    def setupOutputDirectory(self):
        if os.path.exists(self.outputPath):
            shutil.rmtree(self.outputPath)
        os.makedirs(self.outputPath)

    def processAndOrganize(self):
        print("Training Face Recognition Model...")
        self.faceRecognizer.trainModel(self.inputPath)

        personCount = 0
        processedCount = 0

        # Process images and assign IDs
        for filename in os.listdir(self.inputPath):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                imagePath = os.path.join(self.inputPath, filename)

                # Get prediction from face recognition model
                predictedId = self.faceRecognizer.predictImage(imagePath)

                if predictedId is not None:
                    personDir = os.path.join(self.outputPath, f"person_{predictedId}")
                    os.makedirs(personDir, exist_ok=True)

                    # Save the image to the predicted person's directory
                    image = cv2.imread(imagePath)
                    outputPath = os.path.join(personDir, filename)
                    cv2.imwrite(outputPath, image)

                    personCount = max(personCount, predictedId + 1)
                    processedCount += 1

        print(f"Successfully processed {processedCount} images.")
        return personCount

# Main execution function
def main():
    inputPath = "/content/drive/MyDrive/Colab Notebooks/Wobot Assignment/task-1"  # Update with your dataset path
    outputPath = "/content/drive/MyDrive/Colab Notebooks/Wobot Assignment/outputMethod3"  # Update with desired output path

    system = FaceClassificationSystem(inputPath, outputPath)
    system.setupOutputDirectory()

    try:
        personCount = system.processAndOrganize()
        print(f"Total unique individuals detected: {personCount}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
