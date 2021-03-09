from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing_extensions import final
from IMLib.utils import *

import os
from Dataset import Dataset
from GazeGAN import Gaze_GAN
from config.train_options import TrainOptions
from Dataset import save_images

from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import copy

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/nimish/Programs/Hippo/Dlib-Pretrained/shape_predictor_5_face_landmarks.dat')

cap = cv2.VideoCapture(0)
opt = TrainOptions().parse()

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

rects = []
first_done = [False, False]

def capture_frame():
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    global rects
    global first_done

    if(first_done[0] == False):
        rects = detector(gray, 1)
        first_done[0] = True

    if (len(rects) > 0):

        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        scale = 1.22
        w1 = int(w*scale)
        h1 = int(h*scale)
        x1 = x - ((w1-w) // 2)
        y1 = y - ((h1-h) // 2)
        y1 -= 20


        roi = frame[y1:y1+h1, x1:x1+w1]
        roi = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_AREA)
        roi_raw = roi.copy()

        roi_rect = dlib.rectangle(0, 0, 256, 256)

        shape = predictor(roi, roi_rect)
        shape = face_utils.shape_to_np(shape)

        rx, ry = 0, 0
        for (x, y) in shape[0:2]:
            rx += x
            ry += y
        rx //= 2
        ry //= 2
        cv2.circle(roi, (rx, ry), 1, (0, 0, 255), -1)
        
        lx, ly = 0, 0
        for (x, y) in shape[2:4]:
            lx += x
            ly += y
        lx //= 2
        ly //= 2
        cv2.circle(roi, (lx, ly), 1, (0, 255, 0), -1)

        return (roi_raw, roi, lx, ly, rx, ry)

    return None


if __name__ == "__main__":

    # # DATA GENERATION
    # count = 0
    # while(True):

    #     input_data = capture_frame()
        
    #     if(input_data is not None):
    #         img_raw, img, lx, ly, rx, ry = input_data
            
    #         if(first_done[1] == False):
    #             first_done[1] = True
    #             # META DATA
    #             txt_line = f"{count:04d}a {lx} {ly} {rx} {ry}"
    #             with open('./dataset/CustomData/custom_test_fixed.txt', 'a') as the_file:
    #                 the_file.write(txt_line+'\n')
    #             # print(txt_line)
            
    #         ### ALWAYS CLEAR FOLDERS CustomData
    #         # SAVE IMAGE
    #         cv2.imwrite(f"./dataset/CustomData/IMG/{count:04d}a.jpg", img_raw, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    #         cv2.imwrite(f"./dataset/CustomData/IMG/{count:04d}b.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    #         # META DATA
    #         txt_line = f"{count:04d}a {lx} {ly} {rx} {ry}"
    #         with open('./dataset/CustomData/custom_test.txt', 'a') as the_file:
    #             the_file.write(txt_line+'\n')
    #         print(txt_line)

    #         cv2.imshow('IMG', img)
    #         # cv2.imshow('RAW', img_raw)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    #         count += 1
    #         ###output_data = gaze_gan.test_webcam(input_data)

    img = cv2.imread('dataset/CustomData/IMG/0040a.jpg')

    # while(1):
    #     cv2.imshow('IMG', img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    input_data = (img, 92, 107, 175, 103)

    cap.release()
    cv2.destroyAllWindows()
        
    ## RUN TEST
    dataset = Dataset(opt)
    gaze_gan = Gaze_GAN(dataset, opt)
    gaze_gan.build_test_model()

    return_output = gaze_gan.test_webcam(input_data)
    final_output = (return_output + 1.0)
    print(final_output)
    final_output = final_output * 127.5
    final_output = np.array(final_output, dtype=np.uint8)
    print(final_output)
    
    while(1):
        cv2.imshow('IMG', final_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
