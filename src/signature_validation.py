#!/usr/bin/env python

from __future__ import print_function
import cv2
import sys
import numpy as np
import os
import logging
import math
from sklearn.neighbors import KNeighborsClassifier

standard_size = (600,200)
#standard_size = (200,100)
training_dir = '../training_set/Offline_Genuine/'
LOG_DEBUG_ENABLE = False

def log_info(*msg):
    print("[INFO] ", msg)

def log_debug(*msg):
    if LOG_DEBUG_ENABLE:
        print("[DEBUG] ", msg)

def crop(img):
    points = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(points)
    return img[y: y+h, x: x+w]


def rotate_img(img, degree):
    num_rows, num_cols = img.shape[:2]

    d=math.sqrt(num_cols**2 + num_rows**2)
    h=(num_cols*num_cols*1.0)/d

    rotation_matrix = cv2.getRotationMatrix2D((num_cols, 0), degree, 1)

    img_rotation = cv2.warpAffine(img, rotation_matrix, (int(d*2), int(d*2)))
    img_rotation = crop(img_rotation)

    return img_rotation

def get_image_histogram(img, bins=100):
    while len(img) % bins == 0:
        bins -= 1
    size = len(img[0]) // bins
#    hist = np.zeros(bins)
    hist = []
    for b in range(bins):
#        hist[b] = np.count_nonzero(img[:,b*size:(b+1)*size] == 0)
        fg_c = np.count_nonzero(img[:,b*size:(b+1)*size] == 0)
        bg_c = np.count_nonzero(img[:,b*size:(b+1)*size] != 0)
        hist = np.append(hist, fg_c)
        hist = np.append(hist, bg_c)

    fg = np.count_nonzero(img == 0)
    bg = np.count_nonzero(img)

    hist = np.append(hist,fg*1.0/bg )
 #   import pdb; pdb.set_trace()
    return (hist, bins)

def get_histogram_data(img, bins=30):
    img_00 = get_image_histogram(img, bins)
    img_45 = get_image_histogram(rotate_img(img,45), bins)
    img_90 = get_image_histogram(rotate_img(img,60), bins)
    hist=[]
    hist=np.append(hist, img_00[0])
    hist=np.append(hist, img_45[0])
    hist=np.append(hist, img_90[0])
#    import pdb; pdb.set_trace()
    return (hist, bins)

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

def print_intersection_image(img1, img2, bg_char='0', fg_char='1', pure_img=False):
    img = np.zeros(shape=(50,150))
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            img[i,j] = max(img1[i,j], img2[i,j])
    print_img(img, bg_char, fg_char, pure_img)


def preprocess_image(img_path, size = standard_size):
    img = cv2.imread(img_path, 1);
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return img

def display_image(title, img, time=100, is_key_waiting=False, position=(0,0) ):
    x, y = position
    cv2.namedWindow(title)
    cv2.moveWindow(title, x, y)
    cv2.imshow(title, img)
    if is_key_waiting:
        cv2.waitKey(time) & 0XFF

def get_training_set():
    train_paths = []
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
            train_paths.append(img_path)
#    for img in train_datas:
#        display_image("Trainning",img, 10)
    return (train_datas, train_labels, train_paths)

def get_testing_set():
    test_paths = []
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
            test_paths.append(img_path)

#    for img in test_datas:
#        display_image("Testing",img, 10)

    return (test_datas, test_labels, test_paths)

def print_datas(datas):
    for i in datas:
        for j in i:
            print_img(j)

#def standardize_datas(datas, labels, data_paths, hist_bins=100):
def standardize_datas(datas_tuple, hist_bins=100):
#    import pdb; pdb.set_trace()
    datas, labels, data_paths = datas_tuple # Python3 does not support tuple as argument

    new_datas = []
#    new_datas = new_datas.reshape(-1,7500).astype(np.float32)
    for data in datas:
        new_datas.append( get_histogram_data(data, hist_bins)[0])

    new_labels = np.array([int(label) for label in labels])
    return (new_datas, new_labels, data_paths)

def get_data(hist_bins=100):
    (train_datas, train_labels, train_paths) = standardize_datas(get_training_set(), hist_bins)
    (test_datas, test_labels, test_paths) = standardize_datas(get_testing_set(),hist_bins)

    return ((train_datas, train_labels, train_paths), (test_datas, test_labels, test_paths))


def calc_accuracy(result, test_labels):
    ((train_datas, train_labels, train_paths), (test_datas, test_labels, test_paths)) = get_data()
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size

####
    i = 0
    for m in matches:
        log_debug(i," ", m)
        if m == False:
            log_debug("Predition   : ", result[i])
            log_debug("Ground truth: ", test_labels[i]," ", test_paths[i])
        i+=1
####
    return (matches, correct, accuracy)

def validate_sklearn(hist_bins=100, image_size=(600,400), K_neighbors=5):
    standard_size=image_size
    test_imgs = get_testing_set()[0]
    ((train_datas, train_labels, train_paths), (test_datas, test_labels, test_paths)) = get_data(hist_bins)
    clf = KNeighborsClassifier(n_neighbors=K_neighbors, algorithm='auto', n_jobs=-1)
    clf.fit(train_datas,train_labels)

    pred = clf.predict(test_datas)
    matches, correct, accuracy = calc_accuracy(pred, test_labels)

######
    if LOG_DEBUG_ENABLE :

        i = 0
        count = 0
        for m in matches:
            if m == False:
                log_info("Predicted   : ", pred[i])
                log_info("Ground truth: ", test_labels[i])

                j = 0
                if count % 5 == 0:
                    j = count//5
                x = (count%5)*(test_imgs[i].shape[1] + 50)
                y = (count//5)*(test_imgs[i].shape[0] + 70)

                display_image(test_paths[i], test_imgs[i], is_key_waiting=True, position=(x, y))

                count += 1

            i += 1

        while cv2.waitKey(0) != ord('q'):
            log_debug("Press q to quit")
        cv2.destroyAllWindows()
######
#    log_info("Predicted   : ", pred)
#    log_info("Ground truth: ", test_labels)

    return (matches, correct, accuracy)

