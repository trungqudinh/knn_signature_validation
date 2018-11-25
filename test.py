#!/usr/bin/env python

from __future__ import print_function
import cv2
import sys
import numpy as np
import os 
from sklearn.neighbors import KNeighborsClassifier

standard_size = (150,50)
training_dir = '/home/helix/Working/Reporitories/machine_learning/training_set/Offline_Genuine/'

def print_img(img):
	for i in range(len(img)):
		for j in range(len(img[0])):
			if img[i,j] == 0 :
				print('0', end='')
			else:
				print(' ', end='')
			#print("%3d " % bw[i,j], end='')
		print('.')

def get_training_set():
	train_labels = []
	train_datas = []
	for i in range(16):
		train_labels.append(str(i+1).zfill(3))
		train_datas.append([])
		for j in range(10):
			path = training_dir + train_labels[-1] + "_" + str(j+1).zfill(2) + '.PNG'
			if not os.path.isfile(path):
				break
			tmp = cv2.imread(path, 1);
			tmp = cv2.resize(tmp, standard_size, interpolation=cv2.INTER_AREA)
	#		tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
	#		(thresh, tmp) = cv2.threshold(tmp, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
			train_datas[-1].append(tmp)
#		train_datas[-1] = train_datas[-1(]
#	train_labels = np.array(train_labels)
#	train_datas = np.array(train_datas)
	return (train_datas, train_labels)

def get_testing_set():
	test_labels = []
	test_datas = []
	for i in range(16):
		test_labels.append(str(i+1).zfill(3))
		test_datas.append([])
		for j in range(11,21):
			path = training_dir + test_labels[-1] + "_" + str(j+1).zfill(2) + '.PNG'
			if not os.path.isfile(path):
				break
			tmp = cv2.imread(path, 1);
			tmp = cv2.resize(tmp, standard_size, interpolation=cv2.INTER_AREA)
			tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
			(thresh, tmp) = cv2.threshold(tmp, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
			test_datas[-1].append(tmp.ravel())
#		test_datas[-1] = test_datas[-1]
#	test_labels = np.array(test_labels)
#	test_datas = np.array(test_datas)
	return (test_datas, test_labels)

def print_datas(datas):
	for i in datas:
		for j in i:
			print_img(j)

#img = cv2.imread('/home/dqtrung/Working/Window_Sync/Helix/Repository/machine_learning/signature-recognition/data/training/021/forged-01.png', 1)
#img = cv2.imread('/home/dqtrung/Working/Window_Sync/Helix/Repository/machine_learning/Signature-Verification/data/genuine/001001_000.png', 1);
#img = cv2.imread('/home/helix/Working/Reporitories/machine_learning/TrainingSet/Offline_Genuine/001_01.PNG', 1);
#img = cv2.imread('/home/helix/Working/Reporitories/machine_learning/TrainingSet/RealData/trung_01.jpg', 1);


#print(train_labels)
#print(train_datas)

#print(train_datas[0][0])
#print_img(train_datas[0][0])
(train_datas, train_labels) = get_training_set()
(test_datas, test_labels) = get_testing_set()
#print_datas(train_datas)
#model = KNeighborsClassifier()
#model.fit(train_datas, test_labels)
print(train_datas[0])
#print(type(train_datas))
#img = cv2.resize(img, (200, 50), interpolation=cv2.INTER_AREA)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#(thresh, bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
#cv2.imshow('example', bw)
#cv2.waitKey(0)
#
#
##print( img)
##print(gray)
##print(np.matrix(gray))
##print( gray)
#for i in range(len(img)):
#	for j in range(len(img[0])):
#		if bw[i,j] == 0 :
#			print(' ', end='')
#		else:
#			print('0', end='')
#		#print("%3d " % bw[i,j], end='')
#	print('.')
#
#
#
#
##for i in range(len(img)):
##    for j in range(len(img[0])):
##        #print(img[i,j])
##        #print (img[i,j,2],end='')
##        #if (img[i,j] == img[0,0]).all():
##        if (img[i,j,0] < 200 or img[i,j,1] < 200 or img[i,j,2] < 200 ):
##            print('x ', end='')
##        else:
##            print('  ',end='')
##            #print ('{} '.format(img[i,j][0]), end='')
##    print('.')
###print
#
#cv2.destroyAllWindows()
