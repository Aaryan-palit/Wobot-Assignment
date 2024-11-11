# Face Detection and Recognition Project

This repository contains solutions for face detection and recognition using three different methods:

1. **MTCNN + Inception ResNet**
2. **DeepFace**
3. **Custom CNN Model** (if labeled data is available)

The project also includes a Dockerfile to containerize the solution and run it in a consistent environment, as well as a Jupyter Notebook (`Wobot Assignment.ipynb`) that runs all three methods together for testing purposes.

## Contents

- `Solution1.py` - MTCNN + Inception ResNet method implementation
- `Solution2.py` - DeepFace method implementation
- `Solution3.py` - Custom CNN method implementation
- `requirements.txt` - Python dependencies for the project
- `Dockerfile` - File to build a Docker container for running the solution
- `Wobot Assignment.ipynb` - Jupyter Notebook to test all three methods at once
- `output/` - Folder containing output files (including results for each method)

## Requirements

Before running the solutions, ensure that you have the required dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```



Running the Solutions Locally
Option 1: Running with Docker
Build the Docker image using the command:

```bash
docker build -t face-detection-recognition .
Run the Docker container:
```
```bash
docker run -it --rm face-detection-recognition
Option 2: Running Without Docker
Install the required dependencies by running:
```
```bash
pip install -r requirements.txt
```
Run each solution separately:

For MTCNN + Inception ResNet:
```bash
python Solution1.py
```
For DeepFace:
```bash
python Solution2.py
```
For Custom CNN:
```bash
python Solution3.py
```
Testing All Methods Together
To test all three methods in one go, you can use the Jupyter Notebook Wobot Assignment.ipynb. This notebook will execute all three face recognition methods and display the results.

Launch the Jupyter Notebook:

```bash
jupyter notebook Wobot Assignment.ipynb
```
Follow the instructions in the notebook to run and compare the results.

Conclusion
Best Method for Real-Time Use: MTCNN + Inception ResNet
Best Method for Simplicity: DeepFace
Best for Customization: Custom CNN model (if labeled data is available)
