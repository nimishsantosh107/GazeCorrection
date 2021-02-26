from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import copy

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    '/home/nimish/Programs/Hippo/Dlib-Pretrained/shape_predictor_5_face_landmarks.dat')

cap = cv2.VideoCapture(0)


def capture_frame():
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    if (len(rects) > 0):

        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        scale = 1.22
        w1 = int(w*scale)
        h1 = int(h*scale)
        x1 = x - ((w1-w) // 2)
        y1 = y - ((h1-h) // 2)
        y1 -= 20

        frame = cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)

        # cv2.imshow('RAW', frame)

        cv2.imwrite("TEMP.png", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

        _ = input("")

while(True):
    capture_frame()
