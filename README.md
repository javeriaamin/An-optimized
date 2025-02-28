# An optimized features selection approach based on Manta Ray Foraging Optimization (MRFO) method for parasite malaria classification
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
infected_images = load_images_from_folder(path_infected)
uninfected_images = load_images_from_folder(path_uninfected)
labels = np.concatenate((np.ones(len(infected_images)), np.zeros(len(uninfected_images))))
# *Segmentation*
def segment_image(img, clusters=3):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixel_vals = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    kmeans.fit(pixel_vals)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    return segmented_img.reshape(img.shape)
segmented_images = [segment_image(img) for img in infected_images + uninfected_images]
# *Features Extraction and Selection*
def extract_features(model, images):
    model = model(weights='imagenet', include_top=False, pooling='avg')
    images = [cv2.resize(img, (224, 224)) for img in images]
    images = np.array(images) / 255.0
    features = model.predict(images)
    return features
features_eff = extract_features(EfficientNetB0, segmented_images)
features_mob = extract_features(MobileNetV2, segmented_images)
selected_features_eff = MRFO(features_eff)
selected_features_mob = MRFO(features_mob)
# *Feature Fusion and Classification*
fused_features = np.hstack((selected_features_eff, selected_features_mob))
svm = SVC(kernel='linear')
cv_score_1 = cross_val_score(svm, selected_features_eff, labels, cv=10).mean()
cv_score_2 = cross_val_score(svm, selected_features_mob, labels, cv=10).mean()
cv_score_fused = cross_val_score(svm, fused_features, labels, cv=10).mean()

print(f"Accuracy with EfficientNet-B0 features: {cv_score_1 * 100:.2f}%")
print(f"Accuracy with MobileNetV2 features: {cv_score_2 * 100:.2f}%")
print(f"Accuracy with Fused features: {cv_score_fused * 100:.2f}%")




