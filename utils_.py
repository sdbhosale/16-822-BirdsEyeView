import numpy as np
from math import cos, sin, pi

'''
Convert a dictionary into an object
'''
class Decoder(object):
    def __init__(self, d):
        self.d = d
    def __getattr__(self, key):
        return self.d[key]


'''
Find vanishing point on the ground plane
'''
def vanishingPt(cam):
	yaw = cam.yaw * pi / 180
	pitch = cam.pitch * pi / 180

	# Projection of the ray through the center of the image onto the ground
	d = np.array([sin(yaw) / cos(pitch), cos(yaw) / cos(pitch), 0])
	
	# Rotation matrix of yaw and pitch
	Rz = np.array([[cos(yaw), -sin(yaw), 0],
                   [sin(yaw), cos(yaw), 0],
                   [0, 0, 1]])
	Ry = np.array([[1, 0, 0],
                   [0, -sin(pitch), -cos(pitch)],
                   [0, cos(pitch), -sin(pitch)]])
	
	# Intrinsics
	K = np.array([[cam.fx, 0, cam.cx],
				  [0, cam.fy, cam.cy],
				  [0, 0, 1]])

	# Get pixel location of vanishing point
	return K @ Rz @ Ry @ d


'''
Image projection onto the ground plane to get BEV
Reference: Aly, Mohamed. "Real time detection of lane markers in urban streets.",
2008 IEEE intelligent vehicles symposium. IEEE, 2008.
'''
def img2grd(uvLimits, cam):
	uv = np.vstack((uvLimits, np.ones(len(uvLimits[0]))))

	c1 = cos(cam.pitch * pi / 180)
	s1 = sin(cam.pitch * pi / 180)
	c2 = cos(cam.yaw * pi / 180)
	s2 = sin(cam.yaw * pi / 180)

	T = [[-c2 / cam.fx, s1 * s2 / cam.fy, c2 * cam.cx / cam.fx - s1 * s2 * cam.cy / cam.fy - c1 * s2],
		 [s2 / cam.fx, s1 * c2 / cam.fy, -s2 * cam.cx / cam.fx - s1 * c2 * cam.cy /cam.fy - c1 * c2],
		 [0,  c1 / cam.fy, -c1 * cam.cy / cam.fy + s1],
		 [0, -c1 / cam.fy / cam.height, c1 * cam.cy / cam.height / cam.fy - s1 / cam.height]]
	T = np.array(T)
	T *= cam.height

	xy = T @ uv
	xy[0] = xy[0] / xy[-1]
	xy[1] = xy[1] / xy[-1]

	return xy[:2]


'''
BEV image projection to the original perspective image
Reference: Aly, Mohamed. "Real time detection of lane markers in urban streets.",
2008 IEEE intelligent vehicles symposium. IEEE, 2008.
'''
def grd2img(xyGrid, cam):
	P = np.vstack((xyGrid, -cam.height * np.ones(len(xyGrid[0]))))
	
	c1 = cos(cam.pitch * pi / 180)
	s1 = sin(cam.pitch * pi / 180)
	c2 = cos(cam.yaw * pi / 180)
	s2 = sin(cam.yaw * pi / 180)

	T = [[cam.fx * c2 + c1 * s2 * cam.cx, -cam.fx * s2 + c1 * c2 * cam.cx, -s1 * cam.cx],
		 [s2 * (-cam.fy * s1 + c1 * cam.cy), c2 * (-cam.fy * s1 + c1 * cam.cy), -cam.fy * c1 - s1 * cam.cy],
		 [c1 * s2, c1 * c2, -s1]]

	p = T @ P
	p[0] = p[0] / p[-1]
	p[1] = p[1] / p[-1]

	return p[:2]