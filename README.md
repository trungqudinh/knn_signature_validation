# Signature validation using KNN

[Vietnamese version here](Report_VN.md)

# Idea

- Convert each image into histogram, with adjustable bins.
    - From 1 photo, turn 45 degrees and 90 degrees respectively to have 2 other images. 
      Gather up into a bigger image.
    - Cut the picture vertically, the number of pieces cut by the number of bins of the histogram.
    - For each cut of the image, calculate the total value of the pixels containing the image signature.
      This sum will be the value of each bins.
- Convert histogram to vector. This vector will be the input of the algorithm.
- Use the built-in k-nearest neighbors (KNN) algorithm of the Sklearn library to trainning and predict

# The process

- Set parameters when testing.
    - Standardized image file size: default is (600,400)
    - Number of bins of histogram: default is 100 bins.
    - K: default is 3.
- Get trainning and test data, with each image converted to histogram.
    - Pretreatment
        - Retrieve image data
        - Eliminate only noise and standardize image files.
        - Convert images to histograms for trainning and test data.
- Perform trainning and prediction:
    - Sequentially change the input parameters to statistic accuracy.
- Calculate accuracy measurement.

# Trainning and testing

    Perform trainng and predict in different parameters.
    Based on the statistics below, we see the highest accuracy > 96% when using 20 bins histogram for most image sizes and number of K-neighbor.

**Statistic**

    Image size: (600, 200)
    K=5
    bins=[600, 500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

Bin | Accuracy
--- | -------------
600 | 63.9175257732
500 | 67.0103092784
400 | 76.2886597938
300 | 70.1030927835
200 | 69.0721649485
100 | 77.3195876289
 90 | 78.3505154639
 80 | 78.3505154639
 70 | 80.412371134
 60 | 84.5360824742
 50 | 87.6288659794
 40 | 90.7216494845
 30 | 95.8762886598
 20 | 96.9072164948
 10 | 95.8762886598


    Image size: (600, 200)
    K=3
    bins=[600, 500, 400, 300, 200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    
Bin | Accuracy
--- | -------------
600 | 70.1030927835
500 | 69.0721649485
400 | 80.412371134
300 | 72.1649484536
200 | 73.1958762887
100 | 80.412371134
 90 | 83.5051546392
 80 | 86.5979381443
 70 | 87.6288659794
 60 | 89.6907216495
 50 | 90.7216494845
 40 | 93.8144329897
 30 | 97.9381443299
 20 | 98.9690721649
 10 | 94.8453608247


    Image size: (400, 200)
    K=5
    bins=[200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

Bin | Accuracy
--- | -------------
200 | 69.0721649485
100 | 77.3195876289
 90 | 78.3505154639
 80 | 78.3505154639
 70 | 80.412371134
 60 | 84.5360824742
 50 | 87.6288659794
 40 | 90.7216494845
 30 | 95.8762886598
 20 | 96.9072164948
 10 | 95.8762886598


    Image size: (200, 100)
    K=3
    bins=[200, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

Bin | Accuracy
--- | -------------
200 | 73.1958762887
100 | 80.412371134
 90 | 83.5051546392
 80 | 86.5979381443
 70 | 87.6288659794
 60 | 89.6907216495
 50 | 90.7216494845
 40 | 93.8144329897
 30 | 97.9381443299
 20 | 98.9690721649
 10 | 94.8453608247

## Usage

- Go to src/ directory
- python3 main.py

### Note

- Enable debug mode to know wrong prediction

	In src/signature_validation.py
	set LOG_DEBUG_ENABLE = True

## Requirements
- Python 3.6+
- OpenCV 3.2
- Numpy
- Scikit-learn


## Reference

https://github.com/vadi95/Signature-Verification.git

https://github.com/guilhermefloriani/signature-recognition.git

https://github.com/jadevaibhav/Signature-verification-using-deep-learning

https://github.com/luizgh/sigver_wiwd

https://github.com/Aftaab99/OfflineSignatureVerification

https://github.com/beyhangl/Signature_Recognition_DeepLearning

https://github.com/guilhermefloriani/signature-recognition


