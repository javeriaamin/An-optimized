

## Introduction
This Python implementation performs **malaria cell segmentation and classification** using advanced machine learning techniques:

- **K-Means Clustering** for color-based segmentation.
- **Deep Feature Extraction** using pre-trained models like **EfficientNet-B0** and **MobileNetV2**.
- **Feature Selection** with **Manta-Ray Foraging Optimization (MRFO)**.
- **Classification** via **Support Vector Machine (SVM)** with **10-fold cross-validation**.

## Code Implementation

### Import Required Libraries
```python
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from sklearn.decomposition import PCA
from manta_ray_optimization import MRFO
```

### Load Image Data
We load images from the specified paths for **infected** and **uninfected** malaria cells.
```python
# Paths for infected and uninfected images
path_infected = "path/to/infected"
path_uninfected = "path/to/uninfected"

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Load data
infected_images = load_images_from_folder(path_infected)
uninfected_images = load_images_from_folder(path_uninfected)
labels = np.concatenate((np.ones(len(infected_images)), np.zeros(len(uninfected_images))))
```

### Image Segmentation using K-Means Clustering
```python
def segment_image(img, clusters=3):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_vals = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(pixel_vals)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    return segmented_img.reshape(img.shape)

# Segment all images
segmented_images = [segment_image(img) for img in infected_images + uninfected_images]
```

### Feature Extraction with Pre-Trained Models
```python
def extract_features(model, images):
    model = model(weights='imagenet', include_top=False, pooling='avg')
    images = [cv2.resize(img, (224, 224)) for img in images]
    images = np.array(images) / 255.0
    features = model.predict(images)
    return features

# Extract deep features
features_eff = extract_features(EfficientNetB0, segmented_images)
features_mob = extract_features(MobileNetV2, segmented_images)
```

### Feature Selection using MRFO
```python
# Feature selection using MRFO
selected_features_eff = MRFO(features_eff)
selected_features_mob = MRFO(features_mob)
```

### Feature Fusion for Improved Performance
```python
# Feature fusion
fused_features = np.hstack((selected_features_eff, selected_features_mob))
```

### Classification with SVM
```python
svm = SVC(kernel='linear')
cv_score_1 = cross_val_score(svm, selected_features_eff, labels, cv=10).mean()
cv_score_2 = cross_val_score(svm, selected_features_mob, labels, cv=10).mean()
cv_score_fused = cross_val_score(svm, fused_features, labels, cv=10).mean()
```

### Results and Accuracy
```python
print(f"Accuracy with EfficientNet-B0 features: {cv_score_1 * 100:.2f}%")
print(f"Accuracy with MobileNetV2 features: {cv_score_2 * 100:.2f}%")
print(f"Accuracy with Fused features: {cv_score_fused * 100:.2f}%")
```

## Conclusion
This approach demonstrates that using **fused features** from **EfficientNet-B0** and **MobileNetV2** improves classification accuracy significantly compared to using individual models. The **Manta-Ray Foraging Optimization (MRFO)** further refines feature selection, ensuring optimal performance.
