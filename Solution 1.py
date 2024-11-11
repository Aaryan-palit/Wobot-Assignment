import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from facenet_pytorch import MTCNN, InceptionResnetV1

def organizeImages(inputFolder):
    # Create output folder if it doesn't exist
    outputFolder = "output"
    os.makedirs(outputFolder, exist_ok=True)

    # Load all images from the input folder
    images = []
    fileNames = []
    all_embeddings = []  # To store all embeddings
    all_faceLocations = []  # To store face bounding boxes for the images that had faces

    for fileName in os.listdir(inputFolder):
        imgPath = os.path.join(inputFolder, fileName)
        img = cv2.imread(imgPath)
        if img is not None:
            images.append(img)
            fileNames.append(fileName)

    # Initialize MTCNN (Face Detection) and InceptionResnetV1 (Face Recognition)
    mtcnn = MTCNN(keep_all=True)
    model = InceptionResnetV1(pretrained='vggface2').eval()

    # Step 1: Detect faces and extract face embeddings
    embeddings = []
    faceLocations = []  # To store face bounding boxes
    for img in images:
        # Convert image to RGB (MTCNN expects RGB images)
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, probs = mtcnn.detect(rgbImg)
        if boxes is not None:
            # Get face embeddings
            faces = mtcnn(rgbImg)  # Detect and align faces
            for face, box in zip(faces, boxes):
                embedding = model(face.unsqueeze(0))  # Get embedding for the face
                embeddings.append(embedding.detach().cpu().numpy().flatten())

                # Store face bounding box information
                faceLocations.append(box)  # Store box for each detected face

    # Step 2: Cluster the face embeddings using KMeans
    if embeddings:
        embeddings = np.array(embeddings)
        kmeans = KMeans(n_clusters=len(set([str(e) for e in embeddings])), random_state=0).fit(embeddings)

        # Step 3: Organize images into respective folders based on clustering
        personFolders = {}
        for i, label in enumerate(kmeans.labels_):
            personId = label  # Unique ID based on clustering
            if personId not in personFolders:
                personFolders[personId] = []

            # Save the image in the respective folder
            personFolder = os.path.join(outputFolder, f"ID_{personId}")
            os.makedirs(personFolder, exist_ok=True)
            imgPath = os.path.join(inputFolder, fileNames[i])

            # Add image to folder
            newImgPath = os.path.join(personFolder, fileNames[i])
            cv2.imwrite(newImgPath, images[i])

            # Optionally: Annotate images with their ID
            if len(faceLocations) > i:  # Ensure there is a face location for the image
                top, right, bottom, left = map(int, faceLocations[i])
                cv2.putText(images[i], f"{personId}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imwrite(newImgPath, images[i])

    # Step 4: Plot each image separately to show results
    valid_images = len(embeddings)
    plt.figure(figsize=(5, 2 * valid_images))
    for i in range(valid_images):
        plt.subplot(valid_images, 1, i + 1)  # 1 column, N rows
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(f"ID: {kmeans.labels_[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Step 5: Return the total count of individuals
    unique_individuals_count = len(set(kmeans.labels_))
    print(f"Total unique individuals detected: {unique_individuals_count}")

if __name__ == "__main__":
    inputFolder = "task-1"  # Your input folder path
    organizeImages(inputFolder)
