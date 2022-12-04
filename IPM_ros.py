#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import json

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from utils import Decoder
from IPM import IPM

class RealTimeIPM(object):

    def __init__(self):
        camera_key = "test"
        cameras = json.load(open("cameras.json"))
        camParam = Decoder(cameras[camera_key][0])

        ROI = Decoder(cameras[camera_key][1])

        self.resH, self.resW = 640, 960
        uvGrid, scaleH, scaleW = IPM(camParam, ROI, self.resH, self.resW)
        print("Vertical scale: %.2f mm/px" % scaleH)
        print("Horizontal scale: %.2f mm/px" % scaleW)

        # Prepare for bilinear interpolation in the callback
        self.mask = np.zeros(len(uvGrid[0]))
        self.mask = (ROI.left <= uvGrid[0]) & (uvGrid[0] <= ROI.right) \
                    & (ROI.top <= uvGrid[1]) & (uvGrid[1] <= ROI.bottom)

        self.x1 = np.int32(uvGrid[0])
        self.x2 = np.int32(uvGrid[0]+0.5)
        self.y1 = np.int32(uvGrid[1])
        self.y2 = np.int32(uvGrid[1]+0.5)
        self.x = uvGrid[0] - self.x1
        self.y = uvGrid[1] - self.y1

        self.bridge_object = CvBridge()
        rospy.Subscriber("/camera/color/image_rect_raw", Image, self.camera_callback, queue_size=1)
        
    
    def camera_callback(self, data):
        try:
            srcImg = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        height, width = srcImg.shape[:2] 
        
        # Bilinear interpolation
        resImg = np.zeros((self.resH, self.resW, 3))
        x1 = np.clip(self.x1, 0, width - 1)
        x2 = np.clip(self.x2, 0, width - 1)
        y1 = np.clip(self.y1, 0, height - 1)
        y2 = np.clip(self.y2, 0, height - 1)
        for i in range(3):
            resImg[:,:,i] = ((srcImg[y1, x1, i] * (1-self.x) * (1-self.y) + srcImg[y1, x2, i] * self.x * (1-self.y) + 
                            srcImg[y2, x1, i] * (1-self.x) * self.y + srcImg[y2, x2, i] * self.x * self.y) 
                            * self.mask).reshape(self.resH, self.resW)

        cv2.namedWindow('IPM', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('IPM', resImg)
        cv2.resizeWindow('IPM', self.resW, self.resH)
        cv2.waitKey(1)


def main():
    rospy.init_node('IPM', anonymous=True)
    RealTimeIPM()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()