# Team-Electra-Intel-Unnati-2024-

## Overview

The main objective of the "Innovative Monitoring System for TELE ICU Patients Using Video Processing and Deep Learning" project is to enhance patient care in Intensive Care Units (ICUs) by developing an advanced monitoring system. This system leverages video processing and deep learning technologies to continuously observe and analyze patient activities, identify abnormal behaviors, and detect interactions with healthcare professionals and visitors. By providing real-time alerts and detailed analytics, the system aims to improve patient safety, optimize healthcare response times, and support medical staff in delivering high-quality care.
## Table of Contents

- [Data Preparation](#data-preparation)
- [Labeling](#labeling)
- [Model Training](#model-training)
- [Results](#results)
- [Contributors](#contributors)

## Data Preparation

### Step 1: Video to Image Conversion

We converted videos to sequences of images using the OpenCV library in Python. The images are organized into the following categories:

- `patient_alone`
- `patient_abnormal`
- `patient_with_doctor`
- `patient_with_bystander`

#### Script: `videotoimage.py`

This script reads video files and extracts frames at a specified frame rate, saving them into the corresponding folders.
## Labeling
### Step 2: Image labeling
We used labelImg to label the images and save the annotations in XML format. The labeled images are stored in their corresponding folders.

- Tools:
LabelImg: For annotating the images
## Model Training
### Step 3: Training the YOLOv8 Model
We trained a YOLOv8 model for person detection using Google Colab and integrated Roboflow for dataset management. The dataset was split into training, testing, and validation sets.

- Steps:
- Data Organization: Using Roboflow to organize and download the dataset
  #### Code Snippet for Data Download:

```python
!mkdir {HOME}/datasets
%cd {HOME}/datasets

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="J8wu6oMbicB452nma36z")
project = rf.workspace("yolo-rpzka").project("teleicu-br4t2")
version = project.version(6)
dataset = version.download("yolov8")
```
-Training: Executing the training process in Google Colab using the downloaded dataset.
For the full training script, please refer to [intelmodel1.py](scripts/intelmodel1.py).
## Results
### Step 4: Model Output
The model outputs include:

- JPG images with bounding boxes and confidence percentages.
- Confidence curves to visualize the model's performance.
## Contributors

- [Neha Ann Biju](https://github.com/nehaannbiju)
- [Ebin Sebastian](https://github.com/ebin172002)
- [Devatha D](https://github.com/DevathaD)
- [Sreelakshmi J](https://github.com/sreelakshmij56)
- [Nikitha Linto](https://github.com/nikithalinto)



