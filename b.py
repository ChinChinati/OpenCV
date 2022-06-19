import cv2
import numpy as np
import a
import matplotlib.pyplot as plt
import math

img = cv2.imread("CVtask.png", -1)
a1 = cv2.imread("Ha.png")
a2 = cv2.imread("HaHa.png")
a3 = cv2.imread("LMAO.png")
a4 = cv2.imread("XD.png")

aa1 = cv2.imread("Ha.png", -1)
aa2 = cv2.imread("HaHa.png", -1)
aa3 = cv2.imread("LMAO.png", -1)
aa4 = cv2.imread("XD.png", -1)

g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(g,230,255,cv2.THRESH_BINARY_INV)


cont, h = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
l=4
r = 0.001

for c in cont:
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,peri*r,True)
    if len(approx) == l:
        approx = cv2.approxPolyDP(c,peri*0.1,True)
        #cv2.drawContours(img,[approx],-1,(255,0,0),3)
        #All squares selected
        M = cv2.moments(c)
        #print(M)
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])

        
        #find all corners of squares (box)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        bx,by,bw,bh = cv2.boundingRect(c)
        #cv2.rectangle(img,(bx,by),(bx+bw,by+bh),(0,0,0),5)

        #finding squares numbers
        if (img[y,x][0]) == 79 and (img[y,x][1]) == 209 and (img[y,x][2]) == 146 :
            sq=1
            #print(sq)
        if (img[y,x][0]) == 9 and (img[y,x][1]) == 127 and (img[y,x][2]) == 240 :
            sq=2
            #print(sq)
        if np.all(img[y,x]) == 0:
            sq=3
            #print(sq)

        if (img[y,x][0]) == 210 and (img[y,x][1]) == 222 and (img[y,x][2]) == 228 :
            sq=4

        (bottomleft, topleft, topRight, bottomRight) = box
        if sq == 2:
            (topleft, topRight, bottomRight,bottomleft) = box
        topRight =(int(topRight[0]), int(topRight[1]))
        topleft =(int(topleft[0]), int(topleft[1]))
        bottomleft =(int(bottomleft[0]), int(bottomleft[1]))
        bottomRight =(int(bottomRight[0]), int (bottomRight[1]))
        #print(topleft)
        #print(sq)
        
        #sq is square no and box is the corresponding coordinates of corners of that square
        # x y is centre of square

        for ar,aru in zip([a1,a2,a3,a4],[aa1,aa2,aa3,aa4]):
            if sq == a.processAruco(ar)[1]:
                (h, w) = ar.shape[:2]
                angle = a.processAruco(ar)[2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                #transparent rotation
                aru_t = cv2.warpAffine(aru, M, (w, h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
                crp = a.processAruco(ar)[3] #cropping cooordinates
                #topleft,bottomRight,topleft,bottomRight
                aru_t = aru_t[crp[0]:crp[1], crp[2]:crp[3]] 
                ar = a.processAruco(ar)[0]
                #print(sq)

                

                x1 = ((topRight[0]+bottomRight[0])/2.0) 
                y1 = ((topRight[1]+bottomRight[1])/2.0)
                slope = (y1-y)/(x1-x)
                angle = ((math.atan(slope))*180)/(np.pi)

                (h, w) = aru_t.shape[:2]
                scale = 0.74
                
                M = cv2.getRotationMatrix2D(((w/2), (h/2)), -angle, scale)
                aru_t = cv2.warpAffine(aru_t, M, (int(w), int(h)),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
                ar = cv2.warpAffine(ar, M, (int(w), int(h)))

                cor = a.aic(ar)[1]
                cor = cor.reshape((4,2))
                print(cor)
                dx,dy,dw,dh = cv2.boundingRect(cor)
                
                aru_t = aru_t[dx:dx+dw,dy:dy+dh]
                ar = ar[dx:dx+dw,dy:dy+dh]
                aru_t=cv2.resize(aru_t, (int(bw),int(bh)), interpolation= cv2.INTER_LINEAR)
                ar=cv2.resize(ar, (int(bw),int(bh)), interpolation= cv2.INTER_LINEAR)
                
                
                
                
#-----------------------------------------------------------------
                cor = a.aic(ar)[1]
                inversemask = np.zeros(aru_t.shape,np.uint8)
                #print('hehe')
                
                cor = cor.reshape((4,2))
                #print(cor)
                cv2.fillPoly(inversemask,[cor],(255,255,255))
                maski = cv2.bitwise_not(inversemask)

                roi = img[ by:by+bh,bx:bx+bw]
                background = cv2.bitwise_and(roi , maski )
                frontimg = cv2.bitwise_and(aru_t, inversemask)

                aru_t = cv2.add(frontimg,background)
                
                img[by:by+bh,bx:bx+bw] = aru_t
                


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
plt.show()
