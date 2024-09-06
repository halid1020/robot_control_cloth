#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import pyrealsense2 as rs
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from utils import save_depth

class CameraImageRetriever:
    def __init__(self, camera_height=1.0):
        """
        Initialize the camera interface for the Intel RealSense camera.
        """
        rospy.init_node('camera_image_retriever', anonymous=True)

        # Set up RealSense pipeline and config
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Depth post-processing filters
        self.colorizer = rs.colorizer()
        self.hole_filling = rs.hole_filling_filter()
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.2)
        self.temporal.set_option(rs.option.filter_smooth_delta, 24)

        self.camera_height = camera_height
        self.bridge = CvBridge()
        self.intrinsic = None

    def _post_process_depth(self, depth_frame):
        """
        Post-process the depth frame using temporal and hole-filling filters.
        """
        H, W = np.asanyarray(depth_frame.get_data()).shape[:2]
        raw_depth_data = np.asarray(depth_frame.get_data()).astype(float) / 1000
        save_depth(raw_depth_data, filename='raw_depth', directory='./tmp')

        # Post-process depth frame
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)
        depth_data = np.asarray(depth_frame.get_data()).astype(float) / 1000
        depth_data = cv2.resize(depth_data, (W, H))

        # Align and crop depth to color image dimensions
        OW, OH = 0, 0  # Offsets
        depth_data = depth_data[
            max(H // 2 - OH, 0):min(H // 2 + OH, H),
            max(W // 2 - OW, 0):min(W // 2 + OW, W)
        ]
        depth_data = cv2.resize(depth_data, (W, H))

        # Fill blank values with average depth
        blank_mask = (depth_data == 0)
        average_depth = self.camera_height
        depth_data += blank_mask * (average_depth + 0.005)
        depth_data = depth_data.clip(0, average_depth + 0.02)

        return depth_data

    def take_rgbd(self):
        """
        Capture a pair of RGB and depth images from the RealSense camera.
        :return: (color_image, depth_image)
        """
        for _ in range(10):
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not color_frame or not depth_frame:
                rospy.logwarn("Failed to retrieve frames from RealSense camera.")
                return None, None

            self.intrinsic = color_frame.profile.as_video_stream_profile().intrinsics
            depth_image = self._post_process_depth(depth_frame)
            color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    def get_intrinsic(self):
        """
        Return the intrinsic parameters of the RealSense camera.
        """
        return self.intrinsic

    def stop_camera(self):
        """
        Stop the RealSense camera pipeline.
        """
        self.pipeline.stop()

def main():
    """
    Main entry point for capturing images using the CameraImageRetriever.
    """
    camera_retriever = CameraImageRetriever(camera_height=1.0)

    try:
        while not rospy.is_shutdown():
            command = input("Press Enter to capture images or 'q' to quit: ")
            if command.lower() == 'q':
                break

            color_image, depth_image = camera_retriever.take_rgbd()
            if color_image is not None and depth_image is not None:
                cv2.imwrite("captured_color_image.png", color_image)
                cv2.imwrite("captured_depth_image.png", depth_image)

    finally:
        camera_retriever.stop_camera()
        rospy.signal_shutdown("Camera stopped")

if __name__ == "__main__":
    main()
