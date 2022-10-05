# References:
# Video display: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
# Drawing rectangle: https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
# Picking dominant color: https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
# Resizing captured image to be analyzed: https://learnopencv.com/cropping-an-image-using-opencv/#dividing-image-into-patch

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# functions

def find_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        startX = endX

    return bar


cap = cv.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    img = frame[150:300, 150:450]
    img = img.reshape((img.shape[0] * img.shape[1],3))
    
    cv.rectangle(frame, (150,150), (450,300),(0,255,0), 3)
    
    clt = KMeans(n_clusters=3)
    clt.fit(img)
    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)
    
    if ret == True:
        cv.imshow('frame', frame)
        plt.axis("off")
        plt.imshow(bar)
        plt.show()
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv.destroyAllWindows()
