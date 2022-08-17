#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

COCO_KP_CLASSES = (
    "person",
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
KP_NAMES_SHORT={
    0: 'n',
    1: 'ley',
    2: 'rey',
    3: 'lea',
    4: 'rea',
    5: 'ls',
    6: 'rs',
    7: 'lel',
    8: 'rel',
    9: 'lw',
    10: 'rw',
    11: 'lh',
    12: 'rh',
    13: 'lk',
    14: 'rk',
    15: 'la',
    16: 'ra'
}

# segments for plotting
SEGMENTS={
    1: [5, 6],
    2: [5, 11],
    3: [11, 12],
    4: [12, 6],
    5: [5, 7],
    6: [7, 9],
    7: [6, 8],
    8: [8, 10],
    9: [11, 13],
    10: [13, 15],
    11: [12, 14],
    12: [14, 16]
}

if __name__ == '__main__':
    print(type(COCO_KP_CLASSES))
    for i in range(len(COCO_KP_CLASSES)):
        print(COCO_KP_CLASSES[i])

    print(type(KP_NAMES_SHORT))
    for key in KP_NAMES_SHORT:
        print('{}:{}'.format(key,KP_NAMES_SHORT[key]))

    print(type(SEGMENTS))
    for key in SEGMENTS:
        print('{}:{}'.format(key,SEGMENTS[key]))