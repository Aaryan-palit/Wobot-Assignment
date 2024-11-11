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
