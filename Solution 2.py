from deepface import DeepFace
import os
from sklearn.cluster import KMeans
import cv2

def organizeImages(inputFolder):
    outputFolder = "/content/drive/MyDrive/Colab Notebooks/Wobot Assignment/outputMethod2"
    os.makedirs(outputFolder, exist_ok=True)

    images = []
    fileNames = []
    all_embeddings = []

    for fileName in os.listdir(inputFolder):
        imgPath = os.path.join(inputFolder, fileName)
        img = cv2.imread(imgPath)
        if img is not None:
            images.append(img)
            fileNames.append(fileName)

    # Extract embeddings using DeepFace
    for img in images:
        result = DeepFace.represent(img, model_name='Facenet', enforce_detection=False)
        all_embeddings.append(result[0]['embedding'])

    # Clustering embeddings using KMeans
    embeddings = np.array(all_embeddings)
    kmeans = KMeans(n_clusters=len(set([str(e) for e in embeddings])), random_state=0).fit(embeddings)

    # Organizing images into respective folders based on clustering
    for i, label in enumerate(kmeans.labels_):
        personId = label
        personFolder = os.path.join(outputFolder, f"ID_{personId}")
        os.makedirs(personFolder, exist_ok=True)

        imgPath = os.path.join(inputFolder, fileNames[i])
        newImgPath = os.path.join(personFolder, fileNames[i])
        cv2.imwrite(newImgPath, images[i])

    print(f"Total unique individuals detected: {len(set(kmeans.labels_))}")

if __name__ == "__main__":
    inputFolder = "/content/drive/MyDrive/Colab Notebooks/Wobot Assignment/task-1"  # Your input folder path
    organizeImages(inputFolder)
