#!/usr/bin/env python3
from __future__ import print_function

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image

from nav_msgs.msg import Odometry
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_from_matrix

from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import torch

import json
from utils_ import *

def imgmsg_to_cv2(img_msg):
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    img_opencv_rgb = np.ndarray(shape=(img_msg.height, img_msg.width, 3), dtype=dtype, buffer=img_msg.data)
    
    #img_opencv_bgr = cv2.cvtColor(img_opencv_rgb, cv2.COLOR_RGB2BGR)
    #img_opencv_rgb = np.rot90(img_opencv_rgb,1,(1,0))

    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        img_opencv_bgr = img_opencv_bgr.byteswap().newbyteorder()

    return img_opencv_rgb

def cv2_to_imgmsg(cv_image):
    img_msg = Image()

    # scale_percent = 50 # percent of original size
    # width = int(cv_image.shape[1] * scale_percent / 100)
    # height = int(cv_image.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # cv_image = cv2.resize(cv_image, dim, interpolation = cv2.INTER_AREA)
    # cv_image = np.rot90(cv_image,1,(1,0))

    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "rgb8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg

class image_converter:

  def __init__(self, model, device):
    self.image_pub = rospy.Publisher("/inference",Image, queue_size=10)

    self.bridge = CvBridge()

    # self.image_sub = rospy.Subscriber("camera/color/image_raw",Image,self.callback, queue_size=30)
    self.image_sub = rospy.Subscriber("d400/color/image_raw",Image,self.callback, queue_size=30)

    self.odom_sub = rospy.Subscriber("t265/odom/sample", Odometry, self.odom_cb, queue_size=30)

    self.br = tf.TransformBroadcaster()

    self.model = model
    self.device = device


  def callback(self,data):
    #print(data.header.stamp.secs)
    
    try:
      #cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
      cv_image = imgmsg_to_cv2(data)
      input_img = cv_image.copy()
    except CvBridgeError as e:
      print(e)

    bbox,scores = self.run_inference(np.asarray(cv_image))
    
    
    #print(len(bbox),len(scores))

    bb_centroid = []
    for i in range(len(bbox)):
      # print(bbox)
      if(scores[i]>0.5):
        bb = bbox[i]
        cv2.rectangle(cv_image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0,255,0), 5)
        cx = int((bb[0]+bb[2])/2)
        cy = int((bb[1]+bb[3])/2)
        bb_centroid.append([cx,cy])
        # cv2.circle(cv_image, (cx, cy), 5, (0, 0, 255), -1)

    bb_centroid = np.array(bb_centroid)
    
    ipm_img = self.birds_eye(input_img, bb_centroid)

    cv2.imshow('img', ipm_img)
    cv2.waitKey(1)

    try:
      self.image_pub.publish(cv2_to_imgmsg(cv_image))
    #   self.image_pub.publish(cv2_to_imgmsg(ipm_img))
    except CvBridgeError as e:
      print(e)
    

  def run_inference(self, img):
    print(img.shape)
    
    with torch.no_grad():
      results = self.model(img)
      results_np = results.xyxy[0].detach().cpu().numpy()

    # print("*****")
    #print(results.xyxy[0])
    #print("\n")
    #print(results_np)
    #print("\n")

    bbox = []
    scores = []
    for r in results_np:
      bbox.append(r[0:4])
      scores.append(r[4])    

    bbox = np.array(bbox)
    scores = np.array(scores)

    # print(bbox)
    # print("\n")
    # print(scores)
    # print("\n")

    return bbox,scores


  def odom_cb(self, msg):
    orientation_q = msg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion (orientation_list)
    # print("Angles")
    # print(roll*(180/3.142), pitch*(180/3.142), yaw*(180/3.142))

    
  def IPM(self, camParam, ROI, resH, resW):
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


  def birds_eye(self, img, bb_centroid):
    # Adjust the output image size here
    resH, resW = 640, 640

    # img = cv2.imread('Images/IMG00996.jpeg')
    height, width = img.shape[:2] 
    print(height, width)

    # camera_key = "d435i"
    camera_key = "d400"
    cameras = json.load(open("cameras.json"))
    camParam = Decoder(cameras[camera_key][0])

    ROI = Decoder(cameras[camera_key][1])

    # IPM
    uvGrid, scaleH, scaleW = self.IPM(camParam, ROI, resH, resW)
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
    
    print("Pot positions")
    print(bb_centroid.shape)
    for num,pot in enumerate(bb_centroid):
        pot = pot.reshape(-1, 1)
        pot_x, pot_y = img2grd(pot, camParam).ravel()  # x right, y forward, z upward
        print("Pot 3D position: ", pot_x, pot_y, 0)

        pot2cam = np.eye(4)
        pot2cam[:3,-1] = [pot_y/1000, -pot_x/1000, -camParam.height/1000]  # tf assumes x forward, y left whereas
                                                            # (pot_x, pot_y) from IPM assumes x right, y forward
        x, y, z = pot2cam[:3, -1]
        i, j, k, w = tf.transformations.quaternion_from_matrix(pot2cam)
        
        child_frame = "pot" + str(num+1)
        print(child_frame)
        self.br.sendTransform((x,y,z), (i,j,k,w), rospy.Time.now(), child_frame, "t265_pose_frame")

    result = np.zeros((resH, resW, 3))
    result[:,:,0] = resImg[:,:,2]
    result[:,:,1] = resImg[:,:,1]
    result[:,:,2] = resImg[:,:,0]
    return result
  

def get_model(num_classes):
  model = torch.hub.load('./yolov5','custom', path='./saved_models/best.pt', source='local')  
  
  return model


def main(args):

  model = get_model(num_classes=2)
  #model.load_state_dict(torch.load("saved_models/checkpoint-FasterRCNN_Tomato-epoch50.pth"))
  #model.load_state_dict(torch.load("saved_models/checkpoint-FasterRCNN_Tomato-epoch249.pth"))
  
  model.eval()
  
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  print(device)
  model.to(device)

  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter(model, device)


  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main(sys.argv)
