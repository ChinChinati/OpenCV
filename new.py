import cv2
import cvzone
import cv2.aruco as aruco
import numpy as np
import a
import matplotlib.pyplot as plt
import math


img = cv2.imread("CVtask.jpg", -1)
a1 = cv2.imread("Ha.png")
a2 = cv2.imread("HaHa.png")
a3 = cv2.imread("LMAO.png")
a4 = cv2.imread("XD.png")

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
        cv2.drawContours(img,[approx],-1,(255,0,0),3)
        #All squares selected
        M = cv2.moments(c)
        #print(M)
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])

        
        #find all corners of squares (box)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #print(box)

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
            #print(sq)



        #-------------------------------------------------------------
        for ar in [a1,a2,a3,a4]:
          i=0
          if sq == a.processAruco(ar)[1]:

            

            x1 = ((box[3][0]+box[2][0])/2.0) 
            y1 = ((box[3][1]+box[2][1])/2.0)
            cv2.circle(img, (int(x1), int(y1)), 10, (0, 0, 255), -1)
            slope = (y1-y)/(x1-x)

            
            if i== 0:
                aru = cv2.imread("Ha.png",-1)
                print(1)
            if i== 1:
                aru = cv2.imread("HaHa.png",-1)
                print(2)
            if i== 2:
                aru = cv2.imread("LMAO.png",-1)
                print(3)
            if i== 3:
                aru = cv2.imread("XD.png",-1)
                print(4)
            
            angle = a.processAruco(ar)[2]
            (h, w) = ar.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 0.74)
            #print(M)
            aruc = cv2.warpAffine(aru, M, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_TRANSPARENT)
            ar = cv2.warpAffine(ar, M, (w, h))
            angle = ((math.atan(slope))*180)/(np.pi)
            M = cv2.getRotationMatrix2D((w/2, h/2), -angle, 0.74)
            aruc = cv2.warpAffine(aru, M, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_TRANSPARENT)
            ar = cv2.warpAffine(ar, M, (w, h))
            plt.imshow(cv2.cvtColor(ar, cv2.COLOR_BGRA2RGBA))
            plt.show()

            
            #taking new corners after rotation
            (nid,ncor) = a.aic(ar)
            corners= ncor.reshape((4,2)) 
            (topleft, topRight, bottomRight, bottomleft) = corners
            topRight =(int(topRight[0]), int(topRight[1]))
            topleft =(int(topleft[0]), int(topleft[1]))
            bottomleft =(int(bottomleft[0]), int(bottomleft[1]))
            bottomRight =(int(bottomRight[0]), int (bottomRight[1]))
            x = int((topleft[0]+bottomRight[0])/2.0)
            y = int((topleft[1]+bottomRight[1])/2.0)
            cv2.circle(ar, (x,y),5, (255,0,0),-1)
            cv2.putText(ar, str(nid), (x, y-18),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
            p = 15 #padding
            cropped = ar[topleft[0]-p:bottomRight[0]+p, topleft[1]-p:bottomRight[1]+p]
            i+=1
                #print(arco[i].shape)
            '''
                height, width = (arco[i].shape[0] , arco[i].shape[1])
                size = (int(width), int(height))
                center = (width/2,height/2)
                # 回転させたい角度（正の値は反時計回り）
                x1 = ((box[3][0]+box[2][0])/2.0) 
                y1 = ((box[3][1]+box[2][1])/2.0)
                cv2.circle(img, (int(x1), int(y1)), 10, (0, 0, 255), -1)
                slope = (y1-y)/(x1-x)
                angle = ((math.atan(slope))*180)/(np.pi)
                # 拡大比率
                scale = 0.74

                # 回転変換行列の算出
                rotation_matrix = cv2.getRotationMatrix2D(center, -angle, scale)

                # アフィン変換
                an = cv2.warpAffine(arco[i], rotation_matrix,size,flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_TRANSPARENT)
                xx = min(box[0][0],box[1][0],box[2][0],box[3][0])
                yy = min(box[0][1],box[1][1],box[2][1],box[3][1])
                img[0:0+an.shape[0], 0:0+an.shape[1]] = an
                
               

                plt.imshow(cv2.cvtColor(an, cv2.COLOR_BGRA2RGBA))'''

                
        


        # draw the center of the shape on the image
        cv2.circle(img, (x, y), 7, (255, 255, 255), -1)
        cv2.putText(img, "center", (x - 20, y - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        #print(box)
        for co in box:
            cv2.circle(img, (co[0], co[1]), 7, (255, 255, 255), -1)
            cv2.putText(img, (str(co[0])+','+str(co[1])), (co[0] - 20, co[1] - 20),
		    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        img = cv2.drawContours(img,[box],0,(0,0,255),2)




img = cv2.resize(img, (1240, 720))
cv2.imshow("Contour Approximation",img)
cv2.waitKey(0)
cv2.destroyAllWindows()