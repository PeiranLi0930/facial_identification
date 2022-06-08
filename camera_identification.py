#!/usr/bin/env python
""" Facial Detection with Haarcascade

This file is the entire program that detects faces with the usaging of haarcascade methods,
and different haarcascade methods.
Through testing, the haarcascades_frontalface_alt2.xml doesn't fit for video detection with high
image frequency.

"""
__author__ = "Peiran Li"
__contact__ = "Sherlockli930@gmail.com"
__license__ = "MIT"
__status__ = "Production"
__version__ = "0.0.1"

import cv2

'''
This method is the core part of the whole program which implements the facial detection and 
clarifies the facial area with an red rectangle.

:param img the image to be detected
'''
def detect_face(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # load a classifier
    detector = cv2.CascadeClassifier(
        '/Users/lipeiran/opt/anaconda3/pkgs/libopencv-3.4.1-h14a57ad_3/share/OpenCV/haarcascades'
        '/haarcascade_frontalface_default.xml')
    face = detector.detectMultiScale(gray_img, 1.2, 8)
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), color = (0, 0, 255), thickness= 2)
    cv2.imshow('result', img)

video = cv2.VideoCapture(0)
# when the parameter is 0, the system use the default camera

while True:
    ret, frame = video.read()  # read every image in the vedio
    if not ret:
        break
    detect_face(frame)
    if (cv2.waitKey(10) & 0xFF) == 27:  # when the key is pressed, stop this iteration.
        # & is bitwise AND. 27 is the ESC key. or use == ord('q')
        break

video.release()
cv2.destroyAllWindows()