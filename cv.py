import cv2
import cv2.aruco as aruco
import numpy as np
import math
import a 
import cvzone

imgBack = cv2.imread("Ha.png")
a2 = cv2.imread("HaHa.png")
a3 = cv2.imread("LMAO.png")
a4 = cv2.imread("XD.png")
imgFront = cv2.imread("CVtask.png", cv2.IMREAD_UNCHANGED)

imgResult = cvzone.overlayPNG(imgFront, imgBack,[20,20])

cv2.imshow('s', imgResult)
cv2.waitKey(0)
cv2.destroyAllWindows()
