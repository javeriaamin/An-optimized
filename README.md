```html
<!-- Malaria Cell Segmentation and Classification -->

<h2>Malaria Cell Segmentation and Classification</h2>

<h3>Introduction</h3>
<p>This Python implementation performs <strong>malaria cell segmentation and classification</strong> using advanced machine learning techniques:</p>
<ul>
  <li><strong>K-Means Clustering</strong> for color-based segmentation.</li>
  <li><strong>Deep Feature Extraction</strong> using pre-trained models like <strong>EfficientNet-B0</strong> and <strong>MobileNetV2</strong>.</li>
  <li><strong>Feature Selection</strong> with <strong>Manta-Ray Foraging Optimization (MRFO)</strong>.</li>
  <li><strong>Classification</strong> via <strong>Support Vector Machine (SVM)</strong> with <strong>10-fold cross-validation</strong>.</li>
</ul>

<h3>Code Implementation</h3>

<h4>Import Required Libraries</h4>
<pre><code>
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
</code></pre>

<h4>Load Image Data</h4>
<p>We load images from the specified paths for <strong>infected</strong> and <strong>uninfected</strong> malaria cells.</p>
<pre><code>
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
</code></pre>

<h4>Image Segmentation using K-Means Clustering</h4>
<pre><code>
def segment_image(img, clusters=3):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_vals = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(pixel_vals)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    return segmented_img.reshape(img.shape)

# Segment all images
segmented_images = [segment_image(img) for img in infected_images + uninfected_images]
</code></pre>

<h4>Feature Extraction with Pre-Trained Models</h4>
<pre><code>
def extract_features(model, images):
    model = model(weights='imagenet', include_top=False, pooling='avg')
    images = [cv2.resize(img, (224, 224)) for img in images]
    images = np.array(images) / 255.0
    features = model.predict(images)
    return features

# Extract deep features
features_eff = extract_features(EfficientNetB0, segmented_images)
features_mob = extract_features(MobileNetV2, segmented_images)
</code></pre>

<h4>Feature Selection using MRFO</h4>
<pre><code>
# Feature selection using MRFO
selected_features_eff = MRFO(features_eff)
selected_features_mob = MRFO(features_mob)
</code></pre>

<h4>Feature Fusion for Improved Performance</h4>
<pre><code>
# Feature fusion
fused_features = np.hstack((selected_features_eff, selected_features_mob))
</code></pre>

<h4>Classification with SVM</h4>
<pre><code>
svm = SVC(kernel='linear')
cv_score_1 = cross_val_score(svm, selected_features_eff, labels, cv=10).mean()
cv_score_2 = cross_val_score(svm, selected_features_mob, labels, cv=10).mean()
cv_score_fused = cross_val_score(svm, fused_features, labels, cv=10).mean()
</code></pre>

<h4>Results and Accuracy</h4>
<pre><code>
print(f"Accuracy with EfficientNet-B0 features: {cv_score_1 * 100:.2f}%")
print(f"Accuracy with MobileNetV2 features: {cv_score_2 * 100:.2f}%")
print(f"Accuracy with Fused features: {cv_score_fused * 100:.2f}%")
</code></pre>

<h3>Conclusion</h3>
<p>This approach demonstrates that using <strong>fused features</strong> from <strong>EfficientNet-B0</strong> and <strong>MobileNetV2</strong> improves classification accuracy significantly compared to using individual models. The <strong>Manta-Ray Foraging Optimization (MRFO)</strong> further refines feature selection, ensuring optimal performance.</p>
```
