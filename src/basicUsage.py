#!/usr/bin/env python

from __future__ import print_function
import cv2
import sys
import numpy as np
import os 
img = cv2.imread('/home/dqtrung/Working/Window_Sync/Helix/Repository/machine_learning/machine_learning/training_set/Offline_Genuine/001_01.PNG', 1)
img = cv2.resize(img, (200, 50), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
