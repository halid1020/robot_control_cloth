#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import pyrealsense2 as rs

class CameraImageRetriever(Node):
    def __init__(self):
        super().__init__('camera_image_retriever')
        self.bridge = CvBridge()
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_callback,
            10)
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10)
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10)
        self.color_image = None
        self.depth_image = None
        self.intrinsic = None
        time.sleep(2)

    def camera_info_callback(self, msg):
        # Extract the intrinsic matrix from the CameraInfo message
        self.intrinsic = rs.intrinsics()
        self.intrinsic.width = msg.width
        self.intrinsic.height = msg.height
        self.intrinsic.ppx = msg.k[2]
        self.intrinsic.ppy = msg.k[5]
        self.intrinsic.fx = msg.k[0]
        self.intrinsic.fy = msg.k[4]
        self.intrinsic.model = rs.distortion.none  # Assuming no distortion for simplicity
        self.intrinsic.coeffs = [i for i in msg.d]  # Distortion

    def color_callback(self, msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.color_image = np.asarray(self.color_image)
        except Exception as e:
            self.get_logger().error('Failed to convert color image: %s' % str(e))

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            self.depth_image = np.asarray(self.depth_image)
        except Exception as e:
            self.get_logger().error('Failed to convert depth image: %s' % str(e))

    def get_images(self):
        print('getting image')
        rclpy.spin_once(self)
        if self.color_image is not None and self.depth_image is not None:
            cv2.imwrite('color_image.jpg', self.color_image)
           
            self.depth_image = np.asarray(self.depth_image)/1000
            print(self.depth_image)
            self.depth_image = (self.depth_image - np.min(self.depth_image))/(np.max(self.depth_image)-np.min(self.depth_image))
            cv2.imwrite('depth_image.jpg', (self.depth_image*255).astype(np.int8))
            self.get_logger().info('Images saved')
        else:
            self.get_logger().info('Images not available yet')

    def get_intrinsic(self):
        return self.intrinsic
    
    def take_rgbd(self):
        rclpy.spin_once(self)
        if self.color_image is not None and self.depth_image is not None:
            return self.color_image, self.depth_image/1000
        else:
            return None, None

def main(args=None):
    rclpy.init(args=args)
    image_retriever = CameraImageRetriever()

    try:
        while rclpy.ok():
            command = input("Press Enter to capture images or 'q' to quit: ")
            if command.lower() == 'q':
                break
            image_retriever.get_images()
    finally:
        image_retriever.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
