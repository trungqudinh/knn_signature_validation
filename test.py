#!/usr/bin/env python

from __future__ import print_function
import cv2
import sys
#img = cv2.imread('/home/dqtrung/Working/Window_Sync/Helix/Repository/machine_learning/signature-recognition/data/training/021/forged-01.png', 1)
#img = cv2.imread('/home/dqtrung/Working/Window_Sync/Helix/Repository/machine_learning/Signature-Verification/data/genuine/001001_000.png', 1);
img = cv2.imread('/home/dqtrung/Working/Window_Sync/Helix/Repository/machine_learning/trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Genuine/001_01.PNG', 1);

img = cv2.resize(img, (100, 50), interpolation=cv2.INTER_AREA)
cv2.imshow('example', img)
cv2.waitKey(0)
#print img
for i in range(len(img)):
    for j in range(len(img[0])):
        #print(img[i,j])
        #print (img[i,j,2],end='')
        #if (img[i,j] == img[0,0]).all():
        if (img[i,j,0] < 200 or img[i,j,1] < 200 or img[i,j,2] < 200 ):
            print('x ', end='')
        else:
            print('  ',end='')
            #print ('{} '.format(img[i,j][0]), end='')
    print('.')
print

cv2.destroyAllWindows()
