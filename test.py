#!/usr/bin/env python

from __future__ import print_function
import cv2
import sys
import numpy as np
#img = cv2.imread('/home/dqtrung/Working/Window_Sync/Helix/Repository/machine_learning/signature-recognition/data/training/021/forged-01.png', 1)
#img = cv2.imread('/home/dqtrung/Working/Window_Sync/Helix/Repository/machine_learning/Signature-Verification/data/genuine/001001_000.png', 1);
#img = cv2.imread('/home/dqtrung/Working/Window_Sync/Helix/Repository/machine_learning/trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Genuine/001_01.PNG', 1);
img = cv2.imread('/home/helix/Working/Reporitories/machine_learning/TrainingSet/RealData/phuong_01.jpg', 1);

img = cv2.resize(img, (100, 50), interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

(thresh, bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow('example', bw)
cv2.waitKey(0)


#print( img)
#print(gray)
#print(np.matrix(gray))
#print( gray)
for i in range(len(img)):
	for j in range(len(img[0])):
		if bw[i,j] == 0 :
			print('x', end='')
		else:
			print(' ', end='')
		#print("%3d " % bw[i,j], end='')
	print('.')




#for i in range(len(img)):
#    for j in range(len(img[0])):
#        #print(img[i,j])
#        #print (img[i,j,2],end='')
#        #if (img[i,j] == img[0,0]).all():
#        if (img[i,j,0] < 200 or img[i,j,1] < 200 or img[i,j,2] < 200 ):
#            print('x ', end='')
#        else:
#            print('  ',end='')
#            #print ('{} '.format(img[i,j][0]), end='')
#    print('.')
##print

cv2.destroyAllWindows()
