# Signature validation using KNN

## Idea

- Split image into smaller bins at vertical size.
- Each bin will become attribute value of vector.
- Do above step after rotated image
  - Rotate 45 degree
  - Rotate 90 degree

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


