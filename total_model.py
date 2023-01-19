from object_detection import ObjectDetection
import tkinter as tk
from tkinter import *
import cv2 as cv
from PIL import Image
from PIL import ImageTk
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import threading
import sys
import math

veh_model = Sequential()#to extract the features in model
veh_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
veh_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
veh_model.add(MaxPooling2D(pool_size=(2, 2)))
veh_model.add(Dropout(0.25))
veh_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
veh_model.add(MaxPooling2D(pool_size=(2, 2)))
veh_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
veh_model.add(MaxPooling2D(pool_size=(2, 2)))
veh_model.add(Dropout(0.25))
veh_model.add(Flatten())
veh_model.add(Dense(1024, activation='relu'))
veh_model.add(Dropout(0.5))
veh_model.add(Dense(4, activation='softmax'))
veh_model.load_weights('vehicle_model.h5')
cv2.ocl.setUseOpenCL(False)




# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("E:/Mini Project/Required Files/Sample Videos/video3.mp4")

# Initialize count
count = 0
center_points_prev_frame = []
show_text=[0]
global last_frame1

veh_dict = {0: "   Ambulance   ", 1: "Big Vehicle", 2: "  Car  ", 3: " Motorcycle "}

cap = cv.VideoCapture("E:/Mini Project/Required Files/Sample Videos/video3.mp4")



tracking_objects = {}
track_id = 0

while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Point current frame
    center_points_cur_frame = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))
        #print("FRAME N°", count, " ", x, y, w, h)

        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_frame = grey[y:y + h, x:x + w]
        crop_img = np.expand_dims(np.expand_dims(cv2.resize(roi_frame, (48, 48)), -1), 0)
        prediction = veh_model.predict(crop_img)  # predict the veh from the cropped image
        maxindex = int(np.argmax(prediction))

        cv2.putText(frame, (str(object_id) + veh_dict[maxindex]), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)
        #cv.putText(frame, veh_dict[maxindex] + , (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0] = maxindex

    print("Tracking objects")
    print(tracking_objects)


    print("CUR FRAME LEFT PTS")
    print(center_points_cur_frame)


    cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 115:
        break

cap.release()
cv2.destroyAllWindows()
