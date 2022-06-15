import cv2
import cv2.aruco as aruco
import numpy as np
import math

def aic(img):
    arucoDict = aruco.Dictionary_get(aruco.DICT_5X5_250)
    para = aruco.DetectorParameters_create()
    (corners, ids, r) = aruco.detectMarkers(img, arucoDict, parameters =  para)
    return(ids, np.int0(corners))



def processAruco(a):
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
        s1 = (math.sqrt((topleft[1]-y)**2 + (topleft[0]-x)**2))
        cropped = a[topleft[0]-p:bottomRight[0]+p, topleft[1]-p:bottomRight[1]+p]
        return(cropped, markerId, angle, [topleft[0]-p,bottomRight[0]+p,topleft[1]-p,bottomRight[1]+p], [topleft, topRight, bottomRight, bottomleft])
       
