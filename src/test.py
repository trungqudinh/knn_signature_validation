#!/usr/bin/env python

import signature_validation as sv

(train_datas, train_labels, train_paths) = sv.get_training_set()
(test_datas, test_labels, test_paths) = sv.get_testing_set()

#sv.print_img(train_datas[0], pure_img=True)
sv.print_intersection_image(train_datas[0], test_datas[14])
