import torch
import os
import cv2
import time
import dlib
import numpy as np
import h5py
import skin_detector

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

left_increase_ratio = 0.05 #5%
top_increase_ratio = 0.25 #5%

folder_path = 'cohface/'

A = []
M = []
HR = []
BR = []
#for first in [1] :
for first in range(21, 24):
    for second in [0]:
    #for second in [0] :
        capture = cv2.VideoCapture(folder_path + str(first) + "/" + str(second) + "/data.avi")
        next_capture = cv2.VideoCapture(folder_path + str(first) + "/" + str(second) + "/data.avi")
        next_capture.read()
        Non_Faces = []
        frame_num = 0
        while(capture.isOpened()):
            if capture.get(cv2.CAP_PROP_POS_FRAMES) + 1 == capture.get(cv2.CAP_PROP_FRAME_COUNT):
                break
            _, frame = capture.read()
            _, next_frame = next_capture.read()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            if len(rects) != 0:
                rect = rects[0]
                left, right, top, bottom = rect.left(), rect.right(), rect.top(),rect.bottom()
                width = abs(right - left)
                height = abs(bottom - top)
                
                face_left = int(left - (left_increase_ratio/2)*width)
                face_top = int(top - (top_increase_ratio)*height)

                face_right = right
                face_bottom = bottom

                face = frame[face_top:face_bottom,face_left:face_right]
                next_face = next_frame[face_top:face_bottom,face_left:face_right]

                mask = skin_detector.process(face)
                next_mask = skin_detector.process(next_face)

                masked_face = cv2.bitwise_and(face, face, mask=mask)
                next_masked_face = cv2.bitwise_and(next_face, next_face, mask=next_mask)
            else:
                Non_Faces.append(int(capture.get(cv2.CAP_PROP_POS_FRAMES)))
                continue
            frame_num += 1
            face = cv2.resize(face, (36, 36))
            masked_face = cv2.resize(masked_face, (36, 36))
            next_masked_face = cv2.resize(next_masked_face, (36, 36))
            difference = (next_masked_face - masked_face)/(masked_face + next_masked_face + 1)
            if len(np.argwhere(np.isnan(difference))) != 0:
                print(np.argwhere(np.isnan(difference)))
            M.append(difference)
            A.append(face)
        
        capture.release()
        next_capture.release()
        
        if len(Non_Faces) != 0:
            print(Non_Faces)
        data_pulse_ = []
        data_respiration_ = []
        with h5py.File(folder_path + str(first) + "/" + str(second) + "/data.hdf5", "r") as f:
            a_group_key_time = list(f.keys())[2]
            data_time = list(f[a_group_key_time])
            len_data = len(data_time)
            a_group_key_pulse = list(f.keys())[0]
            data_pulse = list(f[a_group_key_pulse])
            a_group_key_respiration = list(f.keys())[1]
            data_respiration = list(f[a_group_key_respiration])
        time = 0
        count = 0
        for i in range(len(data_time)):
            if data_time[i] > time :
                time += 0.05
                if count in Non_Faces:
                    count += 1
                    continue
                count += 1
                data_pulse_.append(data_pulse[i])
                data_respiration_.append(data_respiration[i])
            if len(data_respiration_) == frame_num + 1:
                break
        for i in range(len(data_respiration_) - 1):
            HR.append(data_pulse_[i+1] - data_pulse_[i])
            BR.append(data_respiration_[i+1] - data_respiration_[i])
        print(str(first), str(second), len(A), len(M), len(HR), len(BR))

A = np.asarray(A)
M = np.asarray(M)
HR = np.asarray(HR)
BR = np.asarray(BR)

print(A.shape, M.shape, HR.shape, BR.shape)

np.save("test_A.npy", A)
np.save("test_M.npy", M)
np.save("test_HR.npy", HR)
np.save("test_BR.npy", BR)