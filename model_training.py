#!/usr/bin/env python
''' Create the dataset including the face data in correspondent to the name
This file includes the methods of creating the face-name database (.yml file), which uses the
LBPHFaceRecognizer to train the face-name relationships.
'''
__author__ = "Peiran Li"
__contact__ = "Sherlockli930@gmail.com"
__license__ = "MIT"
__status__ = "Production"
__version__ = "0.0.1"

import os
import cv2
from PIL import Image
import numpy as np

'''
This method gets the face data corresponding to the name. 

:param path: the database path 
:returns face_data: the list of all the face data in the database 
         name_data: the list of all the name data corresponding to the face data in the database
'''
def getImagesAndLables(path):
    face_data = []
    name_data = []
    imagePaths = [os.path.join(path, r) for r in os.listdir(path)]

    face_detector = cv2.CascadeClassifier(
        '/Users/lipeiran/opt/anaconda3/pkgs/libopencv-3.4.1-h14a57ad_3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

    for imagePath in imagePaths:
        # PIL has 9 different forms: '1' - 2valuesIMG, 'L' - grayScaleIMG,
        PIL_image = Image.open(imagePath).convert('L')

        # change the PIL grayScale image to the matrix
        img_matrix = np.array(PIL_image, 'uint8')

        face_matrix = face_detector.detectMultiScale(img_matrix)

        name = int(os.path.split(imagePath)[1].split('.')[0])

        for x, y, w, h in face_matrix:
            name_data.append(name)
            face_data.append(img_matrix[y:y+h, x:x+w])

    print('Name:', name)
    print('Face Matrix:', face_data)
    return face_data, name_data


if __name__ == '__main__':
    path = './data/face'
    faces, names = getImagesAndLables(path)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # training the face data in correspondent to the name
    recognizer.train(faces, np.array(names))
    # trained data
    recognizer.write('trainer/trainer.yml')





