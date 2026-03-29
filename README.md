# Kidney-Stone-Detection-Vision-AI-
Developed a deep learning model using YOLOv8 to detect in CT images, implementing an end-to-end pipeline for data annotation, preprocessing, model training, and testing in a GPU-accelerated Google Colab environment.


---

##  Introduction

Kidney stones are a growing health concern, particularly in regions with limited access to advanced healthcare facilities. Early detection is crucial, yet traditional diagnostic methods rely heavily on CT scans and skilled radiologists, which are not always available in rural or resource-constrained areas.

This project proposes an AI-based solution that uses deep learning and computer vision (Vision AI) to automatically detect kidney stones from CT scan images. The goal is to make diagnosis faster, more accurate, and more accessible, ultimately improving patient outcomes and reducing dependency on manual analysis.

---

## Methodology

The project follows a structured deep learning workflow for medical image analysis. A dataset of annotated CT scan images is collected and preprocessed to ensure quality and consistency. Data augmentation techniques such as rotation and flipping are applied to improve model robustness.

A pre-trained object detection model (YOLOv8) is fine-tuned on the dataset to learn how to identify kidney stones. The model is trained using optimized hyperparameters and evaluated using standard metrics such as precision, recall, and mean Average Precision (mAP). This methodology ensures accurate detection and generalization on unseen data.

---

## System Design

The system is designed as an end-to-end pipeline for automated kidney stone detection. It consists of several key components:

* **Data Input:** CT scan images with corresponding annotations
* **Preprocessing Module:** Image cleaning, resizing, and augmentation
* **Detection Model:** YOLOv8-based deep learning model for object detection
* **Training Environment:** Google Colab with GPU acceleration
* **Output Module:** Visualized results with bounding boxes indicating detected kidney stones



---

##  Experiment & Analysis

The model was trained on a labeled dataset and evaluated using multiple performance metrics. Training results showed a steady decrease in loss values, indicating effective learning.

Key evaluation metrics include:

* **Precision:** Measures accuracy of predictions
* **Recall:** Measures ability to detect all relevant cases
* **mAP (Mean Average Precision):** Evaluates overall detection performance

---

## Code Snippet

```python
# Load and visualize labeled training images
from utils import visualize_dataset
visualize_dataset(train_images_path, train_labels_path)

# Train YOLOv8 model
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='kidney_stone.yaml', epochs=50, batch=16)

# Evaluate model
metrics = model.val()
print(metrics)

# Detect kidney stones on custom image
results = model.predict('test_image.jpg')
results.show()

```



## Hardware and Software Implementation

* **Data**: Kidney stone images, Kaggle, YOLOv8, train/val/test
* **Environment**: Google Colab, GPU, Python, OpenCV, Pandas, Matplotlib, Seaborn
* **Training**: YOLOv8, fine-tuning, hyperparameters, learning rate, epochs, batch size
* **Evaluation**: mAP, IoU, loss curves, metrics, visualization
* **Testing**: Unseen data, detection, localization, validation, deployment





## Team and Credits

Tanjim Subah Alam
Bachelor in Computer Science and Engineering, North South University

Chadni Mandal
Bachelor in Computer Science and Engineering, North South University

Sadia Hassan Chowdhury
Bachelor in Computer Science and Engineering, North South University

Ratul Dey
Bachelor in Computer Science and Engineering, North South University
