import cv2
from cv2 import aruco
import numpy as np

img = cv2.imread("CVtask.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray,30,150)
blank = np.zeros(img.shape,np.uint8)
cont, hier = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
l = 4
r = 0.025
#Find 4 sided shapes
for c in cont:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,peri*r,True)
        print(approx)
        if len(approx) == l:
            cv2.drawContours(blank,[approx],-1,(255,255,255),1)
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
g = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV)
#thresh is the image with outlines of all 4 sided polies
exam2 = img.copy()
cnt, h = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
eps = 0
for c in cnt: 
        peri = cv2.arcLength(c,True)
        cv2.drawContours(exam2,[c],-1,(255,0,0),1)

        approx = cv2.approxPolyDP(c,peri*0.01,True)
        #print(c)
        cv2.drawContours(exam2,[approx],-1,(255,0,0),1)
        cv2.putText(exam2,f"epsilon {eps}",(60,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
cv2.putText(exam2,f"no. of points {len(approx)}",(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
exam2 = cv2.resize(exam2, (1240, 720))  
cv2.imshow("Contour Approximation",exam2)
cv2.waitKey(0)
cv2.destroyAllWindows()





# while True:
#     exam2 = img.copy()
#     peri = cv2.arcLength(cnt,True)
#     eps = cv2.getTrackbarPos("epsilon","Contour Approximation")
#     eps = eps/1000
#     approx = cv2.approxPolyDP(c,peri*eps,True)
#     cv2.drawContours(exam2,[approx],-1,(255,0,0),2)
#     cv2.putText(exam2,f"no. of points {len(approx)}",(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#     cv2.putText(exam2,f"epsilon {eps}",(60,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
#     cv2.imshow("Contour Approximation",exam2)
#     if cv2.waitKey(10) == ord('q'):
#         break
# cv2.destroyAllWindows()
# # thresh = cv2.resize(thresh, (1240, 720))
# # cv2.imshow("examplethresh",thresh)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


[1404  470 1290  747 1566  861 1681  585]
[ 797  419  332  672  586 1137 1051  883]
[1170   76 1170  429 1523  429 1523   76]
[200  34 104 435 505 531 601 129]