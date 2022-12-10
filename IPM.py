import cv2
import numpy as np
import json

from utils_ import *

# IPM
def IPM(camParam, ROI, resH, resW):
    # Calculate vanishing point and adjust top edge of ROI adaptively
    vp = vanishingPt(camParam)
    ROI.top = float(max(int(vp[1]), ROI.top))

    # Limits in the uv plane (unit: px)
    uvLimits = np.array([[vp[0], ROI.top], [ROI.right, ROI.top], 
                        [ROI.left, ROI.top], [vp[0], ROI.bottom]]).T

    # Limites in the xy plane (unit: mm)
    xyLimits = img2grd(uvLimits, camParam)

    # Calculate physical size of each pixel
    xMin, xMax = min(xyLimits[0]), max(xyLimits[0])
    yMin, yMax = min(xyLimits[1]), max(xyLimits[1])
    stepRow = (yMax - yMin) / resH  # mm per vertical px
    stepCol = (xMax - xMin) / resW  # mm per horizontal px

    # Create a 2D grid of x and y values, 
    # with x increasing along the rows and y increasing along the columns
    # The dimension of this grid matches the result BEV image
    x = np.arange(xMin + 0.5 * stepCol, xMax, stepCol)
    y = np.arange(yMax - 0.5 * stepRow, yMin, -stepRow)
    x, y = np.meshgrid(x, y)
    xyGrid = np.array([x.flatten(), y.flatten()])

    # Project back to the pixel space (image plane)
    # to sample the source image with bilinear interpolation
    uvGrid = grd2img(xyGrid, camParam)

    return uvGrid, stepRow, stepCol


if __name__ == '__main__':
    # Adjust the output image size here
    resH, resW = 640, 640

    # Specify input image path here
    imgPath = 'data/IMG00996.jpeg'
    img = cv2.imread(imgPath)
    height, width = img.shape[:2] 

    camera_key = "test"
    cameras = json.load(open("cameras.json"))
    camParam = Decoder(cameras[camera_key][0])

    ROI = Decoder(cameras[camera_key][1])
    
    uvGrid, scaleH, scaleW = IPM(camParam, ROI, resH, resW)
    print("Vertical scale: %.2f mm/px" % scaleH)
    print("Horizontal scale: %.2f mm/px" % scaleW)

    # Mask of ROI
    mask = np.zeros(len(uvGrid[0]))
    mask = (ROI.left <= uvGrid[0]) & (uvGrid[0] <= ROI.right) & (ROI.top <= uvGrid[1]) & (uvGrid[1] <= ROI.bottom)

    # Compute the bilinear interpolated values using the indices and the mask
    x1 = np.int32(uvGrid[0])
    x2 = np.int32(uvGrid[0]+0.5)
    y1 = np.int32(uvGrid[1])
    y2 = np.int32(uvGrid[1]+0.5)
    x = uvGrid[0] - x1
    y = uvGrid[1] - y1

    srcImg = img.astype(float) / 255
    resImg = np.zeros((resH, resW, 3))
    x1 = np.clip(x1, 0, width - 1)
    x2 = np.clip(x2, 0, width - 1)
    y1 = np.clip(y1, 0, height - 1)
    y2 = np.clip(y2, 0, height - 1)
    for i in range(3):
        resImg[:,:,i] = ((srcImg[y1, x1, i] * (1-x) * (1-y) + srcImg[y1, x2, i] * x * (1-y) + 
                        srcImg[y2, x1, i] * (1-x) * y + srcImg[y2, x2, i] * x * y) * mask).reshape(resH, resW)

    # (OPTIONAL) Imageine now we detect a pot at (u, v) in srcImg
    # We can get its 3D location easily as follows
    pot = np.array([580, 850])  # u is column and v is row
    img = cv2.circle(img, (pot[0], pot[1]), 5, (0, 0, 255), -1)
    cv2.imshow('source', img)
    cv2.waitKey()
    pot = pot.reshape(-1, 1)
    pot_x, pot_y = img2grd(pot, camParam).ravel()  # x right, y forward, z upward
    print("Pot 3D position: ", pot_x, pot_y, 0)

    # show the result
    cv2.imshow('result', resImg)
    cv2.waitKey()

    # save image
    cv2.imwrite('%s_ipm.jpg' % imgPath.split('.')[0], resImg * 255)
