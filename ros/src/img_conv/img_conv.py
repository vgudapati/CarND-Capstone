#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml
import time

class ImgConvert(object):
    def __init__(self):
        rospy.init_node('img_conv')

        rospy.Subscriber('/image_color', Image, self.image_cb)
        rospy.Subscriber('/image_raw', Image, self.image_cb)

        self.bridge = CvBridge()

        self.has_image = False
        self.camera_image = None
        self.image_count = 0

        rospy.spin()

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image,"bgr8")
        file = "./site_imgs/site_tl_{0:d}.jpeg".format(self.image_count)
        cv2.imwrite(file, cv_image)            =
        rospy.logwarn("Save image file %s", file)
        self.image_count = self.image_count + 1
    
if __name__ == '__main__':
    try:
        ImgConvert()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start Image Convertor node.')