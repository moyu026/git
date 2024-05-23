# import necessary packages

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import os
import glob
import cv2
import numpy as np

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

imagePaths = getListOfFiles("./dataset/") ## Folder structure: datasets --> sub-folders with labels name
#print(imagePaths)

data = []
labels = []
c = 0 ## to see the progress
for image in imagePaths:

    label = os.path.split(os.path.split(image)[0])[1]
    labels.append(label)

    img = cv2.imread(image)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    data.append(img)
    c = c+1
    print(c)

#print(labels)

# encode the labels as integer
data = np.array(data)
labels = np.array(labels)

le = LabelEncoder()
labels = le.fit_transform(labels)

myset = set(labels)
print(myset)

dataset_size = data.shape[0]
data = data.reshape(dataset_size, -1)

print(data.shape)
print(labels.shape)
print(dataset_size)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
model.fit(trainX, trainY)

print(classification_report(testY, model.predict(testX), target_names=le.classes_))