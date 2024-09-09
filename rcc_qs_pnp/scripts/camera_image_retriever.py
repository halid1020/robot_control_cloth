#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import pyrealsense2 as rs
from utils import save_depth

class CameraImageRetriever:
    def __init__(self, camera_height):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        ### Depth Camera Macros ###
        self.colorizer = rs.colorizer()
        self.hole_filling = rs.hole_filling_filter()
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.2)
        self.temporal.set_option(rs.option.filter_smooth_delta, 24)
        self.camera_height = camera_height

        self.take_rgbd()

    def _post_process_depth(self, depth_frame):
        H, W = np.asanyarray(depth_frame.get_data()).shape[:2]
        raw_depth_data = np.asarray(depth_frame.get_data()).astype(float) / 1000
        save_depth(raw_depth_data, filename='raw_depth', directory='./tmp')

        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)

        depth_data = np.asarray(depth_frame.get_data()).astype(float) / 1000
        depth_data = cv2.resize(depth_data, (W, H))

        ## Align depth to color, to fine tune.
        OW = 0
        OH = 0
        CH = int(H)
        CW = int(W)
        MH = int(H // 2)
        MW = int(W // 2)
        depth_data = depth_data[
            max(MH - CH // 2 + OH, 0): min(MH + CH // 2 + OH, H),
            max(MW - CW // 2 + OW, 0): min(MW + CW // 2 + OW, W)]
        depth_data = cv2.resize(depth_data, (W, H))

        ## get the black ones
        blank_mask = (depth_data == 0)

        average_depth = self.camera_height
        depth_data += blank_mask * (average_depth + 0.005)
        depth_data = depth_data.clip(0, average_depth + 0.02)
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
    

def main():
    rospy.init_node('camera_image_retriever', anonymous=True)
    camera_height = 1.0  # You may replace this with your actual camera height
    image_retriever = CameraImageRetriever(camera_height)

    try:
        while not rospy.is_shutdown():
            command = input("Press Enter to capture images or 'q' to quit: ")
            if command.lower() == 'q':
                break
            image_retriever.take_rgbd()
    finally:
        pass  # No need to destroy nodes or shutdown in rospy

if __name__ == '__main__':
    main()
