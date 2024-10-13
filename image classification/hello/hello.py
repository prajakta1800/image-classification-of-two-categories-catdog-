import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

img_folder = 'image/CatDog'
print("Image Folder:", img_folder)

cat_count = 0
dog_count = 0
for filename in os.listdir(img_folder):
    if filename.startswith('cat'):
        cat_count += 1
    elif filename.startswith('dog'):
        dog_count += 1
print("Cat images:", cat_count)
print("Dog images:", dog_count)

images = []
labels = []
for filename in os.listdir(img_folder):
    # print(f"Processing image: {filename}")
    img = cv2.imread(os.path.join(img_folder, filename))
    # print(f"Image path: {img}")
    img = cv2.resize(img, (224, 224))
    # print(f"Resized image shape: {img.shape}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(f"Image mode: RGB")
    images.append(img)
    if filename.startswith('cat'):
        labels.append(0)  # Label for cat
        # print("Label: Cat (0)")
    elif filename.startswith('dog'):
        labels.append(1)  # Label for dog
        # print("Label: Dog (1)")

images = np.array(images)
labels = np.array(labels)
print("\nDataset Summary:")
print(f"Total images: {len(images)}")
print(f"Total labels: {len(labels)}")
print(f"Image shape: {images.shape[1:]}")
print(f"Label shape: {labels.shape}")

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
print("\nSplit Dataset:")
print(f"Training images: {X_train.shape[0]}")
print(f"Testing images: {X_test.shape[0]}")
print(f"Training labels: {y_train.shape[0]}")
print(f"Testing labels: {y_test.shape[0]}")

X_train = X_train.reshape(-1, 224*224*3)
X_test = X_test.reshape(-1, 224*224*3)
# print(X_train)
# print(X_test)

svm = SVC()
print("SVM Model Created")

svm.fit(X_train, y_train)
print("Model Trained")

y_pred = svm.predict(X_test)
print("Predictions Made",y_pred)

print(classification_report(y_test, y_pred, target_names=['cat', 'dog']))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
















