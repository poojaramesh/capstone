import numpy as np
import cv2

class ColorDescriptor(object):
    def __init__(self, bins):
        # store the number of bins for the 3D histogram
        self.bins = bins

    def describe(self, image):
        # convert the image to the HSV color space and initialize
        # the features used to quantify the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
        
        (h, w) = image.shape[:2]
        fullMask = np.zeros(image.shape[:2], dtype = "uint8")
        cv2.rectangle(fullMask, (0, 0), (w, h), 255, -1)
	
        # update the feature vector
	hist = self.histogram(image, fullMask)
	features.extend(hist)

	# return the feature vector
	return features

    def histogram(self, image, mask):
        # extract a 3D color histogram from the masked region of the
        # image, using the supplied number of bins per channel; then
        # normalize the histogram
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist).flatten()

        # return the histogram
        return hist 
