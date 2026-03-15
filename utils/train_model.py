import os
import numpy as np

from core.pose_classifier import PoseClassifier

data_path = "training_data/latest"

training_data = []
label_data = []

for file_name in os.listdir(data_path):
    if not file_name.endswith(".npy") or "hog_twerking" in file_name:
        continue

    file_path = os.path.join(data_path, file_name)
    data = np.load(file_path)
    label = file_name.split("_latest.npy")[0]

    training_data.extend(data)
    label_data.extend([label] * len(data))

pose_classifier = PoseClassifier()

pose_classifier.train(training_data, label_data)
pose_classifier.save_model()
