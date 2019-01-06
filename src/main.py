#!/usr/bin/env python

import signature_validation as sv
(train_datas, train_labels, train_paths) = sv.get_training_set()
(test_datas, test_labels, test_paths) = sv.get_testing_set()


#bins=[600, 500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
#bins=[600,500, 400, 300, 200, 100]

bins=[100, 50, 20]
print("Standard size of image: ", sv.standard_size)
for bin in bins:
    matches, correct, accuracy =  sv.validate_sklearn(bin)
    print("Accuracy of {} bin histogram: {}".format(bin, accuracy))

