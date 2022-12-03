import cv2
import numpy as np
import json

from utils import *

img = cv2.imread('Images/IMG00996.jpeg')
height, width = img.shape[:2] 

camera_key = "test"
cameras = json.load(open("cameras.json"))
camParam = Decoder(cameras[camera_key][0])

ROI = Decoder(cameras[camera_key][1])

# IPM
vp = vanishingPt(camParam)
ROI.top = float(max(int(vp[1]), ROI.top))

# Limits in the uv plane (unit: px)
uvLimits = np.array([[vp[0], ROI.top], [ROI.right, ROI.top], 
                      [ROI.left, ROI.top], [vp[0], ROI.bottom]]).T

# Limites in the xy plane (unit: mm)
xyLimits = img2grd(uvLimits, camParam)

# =============STILL NEEDS MORE DOCUMENTATION BELOW================
row1 = xyLimits[0, :]
row2 = xyLimits[1, :]
xfMin = min(row1)
xfMax = max(row1)
yfMin = min(row2)
yfMax = max(row2)
xyRatio = (xfMax - xfMin) / (yfMax - yfMin)
outImage = np.zeros((640, 960, 4))
outImage[:,:,3] = 255
outRow, outCol = outImage.shape[:2]
stepRow = (yfMax - yfMin) / outRow
stepCol = (xfMax - xfMin) / outCol
xyGrid = np.zeros((2, outRow * outCol))
y = yfMax - 0.5 * stepRow

for i in range(0, outRow):
    x = xfMin + 0.5 * stepCol
    for j in range(0, outCol):
        xyGrid[0, (i-1) * outCol + j] = x
        xyGrid[1, (i-1) * outCol + j] = y
        x = x + stepCol
    y = y - stepRow

# Back to the pixel space (image plane)
uvGrid = grd2img(xyGrid, camParam)

# mean value of the image
means = np.mean(img) / 255
RR = img.astype(float) / 255

for i in range(0, outRow):
    for j in range(0, outCol):
        ui = uvGrid[0, i*outCol+j]
        vi = uvGrid[1, i*outCol+j]
        #print(ui, vi)
        if ui < ROI.left or ui > ROI.right or vi < ROI.top or vi > ROI.bottom:
            outImage[i, j] = 0.0
        else:
            x1 = np.int32(ui)
            x2 = np.int32(ui+0.5)
            y1 = np.int32(vi)
            y2 = np.int32(vi+0.5)
            x = ui-float(x1)
            y = vi-float(y1)
            outImage[i, j, 0] = float(RR[y1, x1, 0])*(1-x)*(1-y)+float(RR[y1, x2, 0])*x*(1-y)+float(RR[y2, x1, 0])*(1-x)*y+float(RR[y2, x2, 0])*x*y
            outImage[i, j, 1] = float(RR[y1, x1, 1])*(1-x)*(1-y)+float(RR[y1, x2, 1])*x*(1-y)+float(RR[y2, x1, 1])*(1-x)*y+float(RR[y2, x2, 1])*x*y
            outImage[i, j, 2] = float(RR[y1, x1, 2])*(1-x)*(1-y)+float(RR[y1, x2, 2])*x*(1-y)+float(RR[y2, x1, 2])*(1-x)*y+float(RR[y2, x2, 2])*x*y

outImage[-1,:] = 0.0 

# show the result
cv2.imshow('img', outImage)
cv2.waitKey()

# save image
# cv2.imwrite('Images/ipm_test.png', outImage * 255)
