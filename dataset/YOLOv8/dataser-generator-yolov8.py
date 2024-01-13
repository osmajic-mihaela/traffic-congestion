import pandas as pd
from datetime import datetime
import math
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import cv2
from collections import deque
import numpy as np
from ultralytics import YOLO

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

deepsort = None

object_counter = {}
object_counter1 = {}

ids = []
speeds = []
classes = []
directions = []
times = []
frames_id = []
frames_orig = []

line = [(100, 300), (800, 250)]
speed_line_queue = {}


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def estimatespeed(Location1, Location2):
    # Euclidean Distance Formula
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))

    # defining thr pixels per meter
    ppm = 8
    d_meters = d_pixel / ppm
    time_constant = 15 * 3.6

    # distance = speed/time
    speed = d_meters * time_constant

    return int(speed)


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str


def draw_boxes(img, bbox, names, object_cls, frame_id, results, identities=None, offset=(0, 0)):
    # line for counting object
    cv2.line(img, line[0], line[1], (46, 162, 112), 3)

    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
            speed_line_queue[id] = []
        obj_name = names[object_cls[i].item()]
        label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

        # add center to buffer
        data_deque[id].appendleft(center)
        if len(data_deque[id]) >= 2:
            direction = get_direction(data_deque[id][0], data_deque[id][1])
            object_speed = estimatespeed(data_deque[id][1], data_deque[id][0])
            speed_line_queue[id].append(object_speed)
            if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
                cv2.line(img, line[0], line[1], (255, 255, 255), 3)
                if "South" in direction:
                    if obj_name not in object_counter:
                        object_counter[obj_name] = 1
                    else:
                        object_counter[obj_name] += 1
                if "North" in direction:
                    if obj_name not in object_counter1:
                        object_counter1[obj_name] = 1
                    else:
                        object_counter1[obj_name] += 1

        try:
            label = label + " " + str(sum(speed_line_queue[id]) // len(speed_line_queue[id])) + "km/h"
            current_datetime = datetime.now()
            times.append(str(current_datetime.strftime("%Y-%m-%d %H:%M:%S")))
            ids.append(str(id))
            classes.append(obj_name)
            frames_id.append(frame_id)
            frames_orig.append(img)
            sp = str(sum(speed_line_queue[id]) // len(speed_line_queue[id]))
            speeds.append(sp)
            print(
                f'Id:{str(id)}, cls:{obj_name}, speed:{str(sp)}, time:{str(current_datetime.strftime("%Y-%m-%d %H:%M:%S"))}, frame_id:{frame_id}')

            cv2.putText(img, f'{speeds[i]} km/h', (x1, y2), 0, 1, [0, 0, 0], thickness=2,
                        lineType=cv2.LINE_4)
        except:
            pass

        # 4. Display Count in top right corner
        for idx, (key, value) in enumerate(object_counter1.items()):
            cnt_str = str(key) + ":" + str(value)
            cv2.line(img, (width - 500, 25), (width, 25), [85, 45, 255], 40)
            cv2.putText(img, f'Number of Vehicles Entering', (width - 500, 35), 0, 1, [225, 255, 255], thickness=2,
                        lineType=cv2.LINE_AA)
            cv2.line(img, (width - 150, 65 + (idx * 40)), (width, 65 + (idx * 40)), [85, 45, 255], 30)
            cv2.putText(img, cnt_str, (width - 150, 75 + (idx * 40)), 0, 1, [255, 255, 255], thickness=2,
                        lineType=cv2.LINE_AA)

        for idx, (key, value) in enumerate(object_counter.items()):
            cnt_str1 = str(key) + ":" + str(value)
            cv2.line(img, (20, 25), (500, 25), [85, 45, 255], 40)
            cv2.putText(img, f'Numbers of Vehicles Leaving', (11, 35), 0, 1, [225, 255, 255], thickness=2,
                        lineType=cv2.LINE_AA)
            cv2.line(img, (20, 65 + (idx * 40)), (127, 65 + (idx * 40)), [85, 45, 255], 30)
            cv2.putText(img, cnt_str1, (11, 75 + (idx * 40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    return img


def main():
    model = YOLO('yolov8n.pt')

    # Open the video file
    # video_path = "test3.mkv"
    # video_path = "uzas.mp4"
    video_path = "marija.mp4"
    # video_path = "test2.mp4"
    cap = cv2.VideoCapture(video_path)

    # ret, img = cap.read()
    # img = cv2.resize(img,(960, 540))
    # 1080,1920
    # rows, cols, _ = img.shape

    count = 0
    while True:
        ret, img = cap.read()
        img = cv2.resize(img, (960, 540))
        if not ret:
            break

        count += 1
        if count % 6 != 0:
            continue

        # img = img[400:rows,400:1500]
        results = model.track(img, persist=True)
        frame_id = results[0].path
        frame_orig = results[0].orig_img
        objects_id = results[0].boxes.id
        names = results[0].names
        objects_cls = results[0].boxes.cls
        objects_xyxy = results[0].boxes.xyxy

        draw_boxes(frame_orig, objects_xyxy, names, objects_cls, frame_id, results, objects_id)

        cv2.imshow("YOLOv8 Tracking", results[0].plot())

        # cv2.imshow("Img", img)
        key = cv2.waitKey(10)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()