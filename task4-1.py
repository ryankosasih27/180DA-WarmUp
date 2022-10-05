# References:
# BGR and HSV conversion: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
# Video display: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

fourcc = cv.VideoWriter_fourcc(*'DIVX')
# out = cv.VideoWriter('task4-out.avi', fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # with RGB values
        bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        lower_black = np.array([0,0,0])
        upper_black = np.array([55,55,55])
        mask = cv.inRange(bgr, lower_black, upper_black)
        res = cv.bitwise_and(frame,frame, mask= mask)

        # with HSV values
        # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # lower_black = np.array([0,0,0])
        # upper_black = np.array([0,0,55])
        # mask = cv.inRange(hsv, lower_black, upper_black)
        # res = cv.bitwise_and(frame,frame, mask= mask)

        cv.imshow('frame', frame)
        cv.imshow('mask', mask)
        # out.write(res)
        cv.imshow('res', res)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
out.release()
cv.destroyAllWindows()


