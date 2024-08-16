#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import pyrealsense2 as rs
from utils import save_depth

class CameraImageRetriever(Node):
    def __init__(self, camera_height):
        super().__init__('camera_image_retriever')
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        ### Depth Camera Macros ###
        self.colorizer = rs.colorizer()
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 2) #4
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.holes_fill, 2) #3
        self.spatial.set_option(rs.option.filter_magnitude, 5) #5
        self.spatial.set_option(rs.option.filter_smooth_alpha, 1) #1
        self.spatial.set_option(rs.option.filter_smooth_delta, 50) #50
        self.hole_filling = rs.hole_filling_filter()
        self.temporal = rs.temporal_filter()
        self.camera_height = camera_height

        self.take_rgbd()

    def _post_process_depth(self, depth_frame):
        H, W = np.asanyarray(depth_frame.get_data()).shape[:2]
        raw_depth_data = np.asarray(depth_frame.get_data()).astype(float)/1000
        save_depth(raw_depth_data, filename='raw_depth', directory='./tmp')
        #print('raw depth', H, W)
        depth_frame = self.decimation.process(depth_frame)
        depth_frame = self.depth_to_disparity.process(depth_frame)
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.disparity_to_depth.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)

        depth_data = np.asarray(depth_frame.get_data()).astype(float)/1000
        depth_data = cv2.resize(depth_data, (W, H))

        
        

        ## Align depth to color, to fine tune.
        OW = 0 #-14
        OH = 0 #-10
        CH = int(H)
        CW = int(W)
        MH = int(H//2)
        MW = int(W//2)
        depth_data = depth_data[
            max(MH-CH//2+OH, 0): min(MH+CH//2+OH, H), 
            max(MW-CW//2+OW, 0): min(MW+CW//2+OW, W)]
        depth_data = cv2.resize(depth_data, (W, H))

        ## get the blak ones
        blank_mask = (depth_data == 0)



        average_depth = self.camera_height
        depth_data += blank_mask * (average_depth+0.005)
        
        #depth_data = (depth_data + 0.005) - ground_depth + average_depth
        
        
        depth_data = depth_data.clip(0, average_depth+0.02)
        self.depth_img = depth_data.copy()

        return depth_data

    def take_rgbd(self):
        for _ in range(10):
            frames = self.pipeline.wait_for_frames()
        
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            self.intrinsic = color_frame.profile.as_video_stream_profile().intrinsics

            depth_image = self._post_process_depth(depth_frame)
            color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image

    def get_intrinsic(self):
        return self.intrinsic
    

def main(args=None):
    rclpy.init(args=args)
    image_retriever = CameraImageRetriever()

    try:
        while rclpy.ok():
            command = input("Press Enter to capture images or 'q' to quit: ")
            if command.lower() == 'q':
                break
            image_retriever.take_rgbd()
    finally:
        image_retriever.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
