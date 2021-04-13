#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import cv2 as cv
import math


# In[2]:


scale = 1
delta = 0
ddepth = cv.CV_16S

def derivativeSobel(src):
    gray = cv.GaussianBlur(src, (3, 3), 0)
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def derivativeScharr(src):
    gray = cv.GaussianBlur(src, (3, 3), 0)
    grad_x = cv.Scharr(gray, ddepth, 1, 0)
    grad_y = cv.Scharr(gray, ddepth, 0, 1)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad


# In[3]:


def houghP(src):
    dst = cv.Canny(src, 50, 200, None, 3)
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    max = -1
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            if max<l[1]:
                max = l[1]
            if max< l[3]:
                max = l[3]            
            
    return max


# In[4]:


def hough(src):
    dst = cv.Canny(src, 50, 200, None, 3)
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    lines = cv.HoughLines(src,1,np.pi/180,200)
#     lines = cv.HoughLines(dst, 1, np.pi/180.0, 100, np.array([]), 0, 0)
    if lines is not None:
        for line in lines:
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv.line(src,(x1,y1),(x2,y2),(0,0,255),5, cv.LINE_AA)
    return src


# In[5]:


def hough2(src,prev):
    dst = cv.Canny(src, 50, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    
#     lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    lines = cv.HoughLines(dst, 1, np.pi / 180,80)
    x1A=[]
    x2A=[]
    y1A=[]
    y2A=[]
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            x1A.append(int(x0 + 1000*(-b)))
            y1A.append(int(y0 + 1000*(a)))
            x2A.append(int(x0 - 1000*(-b)))
            y2A.append(int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 1, cv.LINE_AA)
        return x1A,x2A,y1A,y2A,cdst
    return x1A,x2A,y1A,y2A,src


# In[30]:


def medialAxis(src):
    x1A,x2A,y1A,y2A,img = hough2(src,None)
    max = houghP(src)
    src = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
    coord = []
    if len(x1A)>0:
        x1=int(np.median(x1A))
        x2=int(np.median(x2A))
        y1=int(np.median(y1A))
        y2=int(np.median(y2A))
        if(y2>y1):
            x2+=(x2-x1)*(max-y2)/(y2-y1)
            x2=int(x2)
            y2=max
        else:
            x1+=(x1-x2)*(max-y1)/(y1-y2)
            x1=int(x1)
            y1=max
        cv.line(src, (x1,y1), (x2,y2), (0,0,255), 3, cv.LINE_AA)
        print(x1,y1,x2,y2)
        coord += [x1]
        coord += [y1]
        coord += [x2]
        coord += [y2]
#     if len(coord) == 0:
#         coord += [0]
#         coord += [0]
#         coord += [0]
#         coord += [0]
    return src, coord


# In[ ]:


cap = cv.VideoCapture('vid/9.mp4')
fgbg = cv.createBackgroundSubtractorMOG2()
hTransform = None

fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('9.avi',fourcc, 15, (1920, 1080))
# out = cv.VideoWriter('output.avi', -1, 20.0, (640,480))
prevavg = None
prev1 = 0
prev2 = 0
# prev3 = 0
oldmed = None
prevCord = None
count = 0
while(1):
    ret, frame = cap.read()
    if ret == True:        
        fgmask = fgbg.apply(frame)
        kernel = np.ones((5,5),np.uint8)
        opening = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
        derScharr = derivativeScharr(closing)
#         hTransform = houghP(derScharr)
        med, coord =medialAxis(derScharr)
        if len(coord)==0:
            if oldmed is not None:
                med = oldmed
                coord = prevCord
                ans = cv.line(frame, (coord[0],coord[1]), (coord[2],coord[3]), (0,0,255), 3, cv.LINE_AA)
            else:
                ans = frame
        else:
            oldmed = med
            prevCord = coord
            ans = cv.line(frame, (coord[0],coord[1]), (coord[2],coord[3]), (0,0,255), 3, cv.LINE_AA)
        
#         if hTransform is None:
#             hTransform = derScharr
#         lines=getLine(lines)
        #cv.imshow('skel',skel(fgmask))
#         frame+=hTransform
#         cv.imshow('corner',corner(derScharr))
#         den = coord[2] - coord[0]
#         count+=1
#         if den == 0:
#             den = 0.001
#         slope = (coord[3] - coord[1]) / den
#         if count>20:            
#             if( abs(slope - prevavg) > 2):
#                 slope = prevavg
#                 med = oldmed
#             else:
# #                 prev3 = prev2
#                 prev2 = prev1 
#                 prev1 = slope
#                 prevavg = (prev1 + prev2)/2
# #                 preavg = prev1
#                 oldmed = med
#         else:
# #             prev3 = prev2
#             prev2 = prev1 
#             prev1 = slope
#             prevavg = (prev1 + prev2)/2
# #             preavg = prev1
#             oldmed = med
        cv.imshow('frame',frame)
        derScharr = cv.cvtColor(derScharr, cv.COLOR_GRAY2BGR)
        cv.imshow('ans',ans)
        cv.imshow('med',med)
        out.write(ans) 
        print("written")

#         cv.imshow('skel',skel(frame))
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
cap.release()
out.release()
cv.destroyAllWindows()


# In[ ]:





# In[ ]:





