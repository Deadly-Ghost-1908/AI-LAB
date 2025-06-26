import cv2
import os
import numpy as np
recognizer = cv2.face.LBPHFaceRecognizer_create()
dataset_path = "dataset"
labels, faces, label_names = [], [], []
for label, persons in enumerate(os.listdir("dataset")):
    label_names.append(persons)
    for img_name in os.listdir(f"dataset/{persons}"):
        img = cv2.imread(f"dataset/{persons}/{img_name}", cv2.IMREAD_GRAYSCALE)
        if img is not None:
            faces.append(cv2.resize(img, (160,160)))
            labels.append(label)
faces_array = np.array(faces)
labels_array = np.array(labels)
recognizer.train(faces_array, labels_array)
test_img_path = "test.jpg"
test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
if test_img is None:
    print("Error: Could Not Able To Read the Test Image.... Please Check Your Path & Try Again")
    exit()
resized_test_img = cv2.resize(test_img, (160,160))
label , confidence = recognizer.predict(resized_test_img)
Predicted_Name = label_names[label]
print(f"Predicted Name : {Predicted_Name}")