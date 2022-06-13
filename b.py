import cv2
import cv2.aruco as aruco
import numpy as np
import math
import matplotlib.pyplot as plt
def aic(img):
    arucoDict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    para = aruco.DetectorParameters_create()
    (corners, ids, r) = aruco.detectMarkers(img, arucoDict, parameters =  para)
    return(ids, np.int0(corners))



bg_img = cv2.imread('CVtask.png', -1)
a = cv2.imread('Ha.png', -1)
(id,cor) = aic(a)
ids = id.flatten()
 #print(ids)
for (markerCorner, markerId) in zip(cor,ids):
        corners= markerCorner.reshape((4,2)) 
        (topleft, topRight, bottomRight, bottomleft) = corners
        topRight =(int(topRight[0]), int(topRight[1]))
        topleft =(int(topleft[0]), int(topleft[1]))
        bottomleft =(int(bottomleft[0]), int(bottomleft[1]))
        bottomRight =(int(bottomRight[0]), int (bottomRight[1]))

        cx = int((topleft[0]+bottomRight[0])/2.0)
        cy = int((topleft[1]+bottomRight[1])/2.0)
        x1 = ((topRight[0]+bottomRight[0])/2.0) 
        y1 = ((topRight[1]+bottomRight[1])/2.0)
        slope = (y1-cy)/(x1-cx)
        angle = ((math.atan(slope))*180)/(np.pi)
        #print (angle)

        #rotation
        (h, w) = a.shape[:2]
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
        #print(M)
        a = cv2.warpAffine(a, M, (w, h))

        #taking new corners after rotation
        (nid,ncor) = aic(a)
        corners= ncor.reshape((4,2)) 
        (topleft, topRight, bottomRight, bottomleft) = corners
        topRight =(int(topRight[0]), int(topRight[1]))
        topleft =(int(topleft[0]), int(topleft[1]))
        bottomleft =(int(bottomleft[0]), int(bottomleft[1]))
        bottomRight =(int(bottomRight[0]), int (bottomRight[1]))
        x = int((topleft[0]+bottomRight[0])/2.0)
        y = int((topleft[1]+bottomRight[1])/2.0)
        cv2.circle(a, (x,y),5, (255,0,0),-1)
        cv2.putText(a, str(markerId), (x, y-18),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
        p = 15 #padding
        cropped = a[topleft[0]-p:bottomRight[0]+p, topleft[1]-p:bottomRight[1]+p]
        img = cropped.copy()
        height, width = (img.shape[0] , img.shape[1])
        size = (width, height)
        center = (width/2,height/2)

        # 回転させたい角度（正の値は反時計回り）
        angle = 45.0

        # 拡大比率
        scale = 0.5

        # 回転変換行列の算出
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        # アフィン変換
        img_dst = cv2.warpAffine(img, rotation_matrix, size,
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_TRANSPARENT)

        resize = ( int(img_dst.shape[1]/2), int(img_dst.shape[0]/2) )
        img_dst = cv2.resize(img_dst, resize)

        print(type(img_dst))
        print(img_dst.shape)

        plt.imshow(cv2.cvtColor(img_dst, cv2.COLOR_BGRA2RGBA))
        plt.show()

        x_offset = int((bg_img.shape[1] - img_dst.shape[1])/2)
        y_offset =  int((bg_img.shape[0] - img_dst.shape[0])/2)

        y1, y2 = y_offset, y_offset +img_dst.shape[0]
        x1, x2 = x_offset, x_offset + img_dst.shape[1]

        alpha_s = img_dst[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            bg_img[y1:y2, x1:x2, c] = (alpha_s * img_dst[:, :, c] + alpha_l * bg_img[y1:y2, x1:x2, c])


        plt.imshow(cv2.cvtColor(bg_img, cv2.COLOR_BGRA2RGBA))
        plt.show()



            



        






#-----------------------------------------------

#putting aruco on the image
#(x,y) = (100,100)#points to put the smaller image on bigger one
#exam[x:x+a1.shape[0], y:y+a1.shape[1]] = a1


