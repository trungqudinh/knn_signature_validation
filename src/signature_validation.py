#!/usr/bin/env python

from __future__ import print_function
import cv2
import sys
import numpy as np
import os 
import logging

standard_size = (150,50)
training_dir = '../training_set/Offline_Genuine/'

def print_img(img, bg_char='0', fg_char='1', pure_img=False):
	for i in range(len(img)):
		for j in range(len(img[0])):
                    if pure_img:
			print("%3d " % img[i,j], end='')
                    else:
			if img[i,j] == 0 :
				print(fg_char, end='')
			else:
				print(bg_char, end='')

		print('.')
def preprocess_image(img_path, size = standard_size):
	img = cv2.imread(img_path, 1);
	img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	return img

def get_training_set():
	train_labels = []
	train_datas = []
	for i in range(16):
		for j in range(10):
                        label = str(i+1).zfill(3)
			img_path = training_dir + label + "_" + str(j+1).zfill(2) + '.PNG'
			if not os.path.isfile(img_path):
				break
		        train_labels.append(label)
			img = preprocess_image(img_path)
			train_datas.append(img)
	return (train_datas, train_labels)

def get_testing_set():
	test_labels = []
	test_datas = []
	for i in range(16):
		for j in range(11,21):
                        label = str(i+1).zfill(3)
			img_path = training_dir + label + "_" + str(j+1).zfill(2) + '.PNG'
			if not os.path.isfile(img_path):
				break
		        test_labels.append(label)
			img = preprocess_image(img_path)
			test_datas.append(img)
	return (test_datas, test_labels)

def print_datas(datas):
	for i in datas:
		for j in i:
			print_img(j)

def log_info(msg):
    print("[INFO] " + msg);
#print_img(train_datas[0][0])
#print_datas(train_datas)
#model = KNeighborsClassifier()
#model.fit(train_datas, test_labels)
#print(train_datas[0])
#cv2.destroyAllWindows()
def standardize_datas((datas, labels)):
    new_datas = np.array(datas)
    new_datas = new_datas.reshape(-1,7500).astype(np.float32)

    new_labels = np.array([int(label) for label in labels])
    return (new_datas, new_labels)
def get_data():
    (train_datas, train_labels) = standardize_datas(get_training_set())
    (test_datas, test_labels) = standardize_datas(get_testing_set())

    return ((train_datas, train_labels), (test_datas, test_labels))

def validate(size = standard_size, noNeighbours = 5):

    standard_size = size
    (train_datas, train_labels) = standardize_datas(get_training_set())
    (test_datas, test_labels) = standardize_datas(get_testing_set())

    log_info("Create KNN")
    knn = cv2.KNearest()
    log_info("Trainning")
    knn.train(train_datas,train_labels)
    log_info("Validating")
    ret,result,neighbours,dist = knn.find_nearest(test_datas,k=5)
    log_info("Done")
    return (ret,result,neighbours,dist,test_labels)

def calc_accuracy(result, test_labels):
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    return (matches, correct, accuracy)


