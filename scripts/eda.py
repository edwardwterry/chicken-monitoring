#! /usr/bin/python

import sys
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

# class image_converter:

#   def __init__(self):
#     self.bridge = CvBridge()
#     self.color_sub = rospy.Subscriber("color/camera/image_raw_throttle", Image, self.color_clbk)
#     self.color_pub = rospy.Publisher("color/camera/processed", Image, queue_size=1)
#     self.therm_sub = rospy.Subscriber("thermal/camera/image_raw_throttle", Image, self.therm_clbk)
#     self.therm_pub = rospy.Publisher("thermal/camera/processed", Image, queue_size=1)
#     self.dx = 470
#     self.dy = 0
#     self.wc = 1920
#     self.hc = 1080
#     self.wt = 80
#     self.ht = 60
#     self.wtc = 1660 - self.dx
#     self.htc = 955 - self.dy
#     therm_size = (80, 60)
#     therm_magnif = 5
#     self.blur_window = 3
#     self.intensity_threshold = 150
#     self.color_contours = []
#     self.processed_color = None
#     self.contours_therm = None

#   def color_clbk(self,data):
#     try:
#       frame_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
#     except CvBridgeError as e:
#       print(e)
    
#     frame_color = cv.rectangle(frame_color, (self.dx, self.dy), (self.dx + self.wtc, self.dy + self.htc), (0, 255, 0), 5)
#     cv.drawContours(frame_color, self.color_contours, -1, (255, 0, 0), 1)

#     self.color_pub.publish(self.bridge.cv2_to_imgmsg(frame_color, "bgr8"))

#   def therm_clbk(self,data):
#     try:
#       frame_therm = self.bridge.imgmsg_to_cv2(data, "mono8")
#     except CvBridgeError as e:
#       print(e)

#     frame_therm = cv.blur(frame_therm, (self.blur_window, self.blur_window))
#     _, thresh = cv.threshold(frame_therm, self.intensity_threshold, 255, cv.THRESH_BINARY)
#     contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
#     frame_therm = cv.cvtColor(frame_therm, cv.COLOR_GRAY2BGR)
#     cv.drawContours(frame_therm, contours, -1, (255, 0, 0), 1)
#     # self.color_contours = self.transform_contours(contours)
#     self.therm_pub.publish(self.bridge.cv2_to_imgmsg(frame_therm, "bgr8"))

#   def transform_contours(self, therm_contours):
#     rot = np.identity(3)
#     scale = np.identity(3)
#     trans = np.identity(3)
#     scale[0,0] = float(self.wt) / float(self.wtc)
#     scale[1,1] = float(self.ht) / float(self.htc)
#     trans[0,2] = -float(self.dx)
#     trans[1,2] = -float(self.dy)  
#     print scale, trans, rot  
#     T_color_therm = np.dot(trans, np.dot(scale, rot))
#     print T_color_therm
#     T_color_therm = np.linalg.inv(T_color_therm)
#     color_contours = []
#     for contour in therm_contours:
#         color_contour = []
#         for point in contour:
#             print point
#             print type(point)
#             np.append(point, 1.0) # to make homogenous
#             color_contour.append(np.dot(T_color_therm * point))
#         color_contours.append(np.array(color_contour))
#     return np.array(color_contours)


# def main(args):
#   rospy.init_node('eda', anonymous=True)
#   ic = image_converter()
#   try:
#     rospy.spin()
#   except KeyboardInterrupt:
#     print("Shutting down")
#   cv.destroyAllWindows()

# if __name__ == '__main__':
#     main(sys.argv)

# def in_bounds(x, y):
#     return x >= xmin and y >= ymin and x <= xmax and y <= ymax

# def get_therm_pixel_from_color_pixel(x_c, y_c):
#     if not in_bounds(x_c, y_c):
#         return None
#     else:
#         res = np.dot(T_color_therm, np.array([x_c, y_c, 1]))
#         return res[0,2], res[1,2]    

br = CvBridge()

blur_window = 3
intensity_threshold = 150 # 150
kernel = np.ones((3,3),np.uint8)
color_k = 1
dx = int(400 * color_k)
dy = int(0 * color_k)
wc = 1920 * color_k
hc = 1080 * color_k
wt = 80
ht = 60
wtc = int((1660 * color_k - dx))
htc = int((940 * color_k - dy))

scale = np.identity(3)
trans = np.identity(3)
scale[0,0] = wtc / wt
scale[1,1] = htc / ht
trans[0,2] = dx
trans[1,2] = dy
T_color_therm = np.dot(trans, scale)

times = []
average_areas = []
contours_therm = None
processed_therm = None

def process_thermal(msg):
    '''
    Convert from ROS to cv2 format
    Apply threshold on image to generate binary mask
    Apply morphological operations to reduce noise
    Calculate contours of binary masks
    '''
    im = br.compressed_imgmsg_to_cv2(msg)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.blur(im, (blur_window, blur_window))
    _, thresh = cv2.threshold(im, intensity_threshold, 255, cv2.THRESH_BINARY)
    erode = cv2.erode(thresh,kernel,iterations = 1)    
    dilate = cv2.dilate(erode,kernel,iterations = 1)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    return dilate, contours

def process_color(msg):
    '''
    Convert from ROS to cv2 format
    '''
    img = br.compressed_imgmsg_to_cv2(msg)
    width = int(img.shape[1] * color_k)
    height = int(img.shape[0] * color_k)
    dim = (width, height)    
    img = cv2.resize(img, dim)
    return img

def therm_clbk(msg):
    '''
    Process received image
    Calculate properties of blobs
    '''
    global contours_therm
    global processed_therm
    processed_therm, contours_therm = process_thermal(msg)
#     set_c(contours_therm)
    therm_pub.publish(br.cv2_to_imgmsg(processed_therm))
    
def color_clbk(msg):
    global contours_therm
    global processed_therm
    processed_color = process_color(msg)
    size = processed_color.shape
    
    mask = np.zeros((size[0], size[1]))
    if contours_therm is not None and processed_therm is not None:
        for c in contours_therm:
            for p in c:
                p = np.append(p, 1.0) # make homogeneous
                res = np.dot(T_color_therm, p)
                res = [int(x) for x in res]
                # mask[res[1], res[0]] = 1
                # mask = cv2.dilate(mask, np.ones((1,1),np.uint8), iterations=1)
#         mask_inv = cv2.bitwise_not(mask)
#         processed_color = cv2.bitwise_and(processed_color,processed_color,mask = mask_inv)
        # processed_color[np.where(mask == 1)] = [0, 255, 0]  
#         plt.imshow(processed_color)
        processed_color = cv2.rectangle(processed_color, (dx, dy), (dx + wtc, dy + htc), (0, 255, 0), 5)

        color_pub.publish(br.cv2_to_imgmsg(processed_color, encoding='bgr8'))    
therm_pub = rospy.Publisher("thermal/camera/processed", Image, queue_size=1)
color_pub = rospy.Publisher("color/camera/processed", Image, queue_size=1)
therm_sub = rospy.Subscriber("thermal/camera/image_raw_throttle/compressed", CompressedImage, therm_clbk)
color_sub = rospy.Subscriber("color/camera/image_raw_throttle/compressed", CompressedImage, color_clbk)

def main(args):
  rospy.init_node('eda', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")  

if __name__ == '__main__':
    main(sys.argv)    