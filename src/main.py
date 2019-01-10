#!/usr/bin/env python

import signature_validation as sv
(train_datas, train_labels, train_paths) = sv.get_training_set()
(test_datas, test_labels, test_paths) = sv.get_testing_set()


def run(img_size, K, bins):
    print("    Image size: {}\n    K={}\n    bins={}\n".format(img_size, K, bins))
    for bin in bins:
        matches, correct, accuracy =  sv.validate_sklearn(bin, img_size, K)
        print("{} | {}".format(bin, accuracy))


size=(600,200)
k=5
bins=[600, 500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
run(size, k, bins)

size=(600,200)
k=3
bins=[600, 500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
run(size, k, bins)

size=(400,200)
k=5
bins=[200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
run(size, k, bins)

size=(200, 100)
k=3
bins=[200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
run(size, k, bins)
