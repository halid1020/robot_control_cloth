#!/usr/bin/env python

from ur_msgs.srv import SetIO
import rospy
import time


class ActiveGripperControl:
    def __init__(self):
        rospy.init_node('client_test', anonymous=True)
        self.cli = rospy.ServiceProxy('/io_and_status_controller/set_io', SetIO)

        # Wait for the service to be available
        rospy.wait_for_service('/io_and_status_controller/set_io')
        self.req = SetIORequest()

    def open(self):
        self.req.fun = 1  # Use the constant for setting digital output
        self.req.pin = 16  # Specify the pin for the tool digital output
        self.req.state = 0.0  # Set the desired state (on/off)
        try:
            self.cli(self.req)
            time.sleep(0.5)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def grasp(self):
        self.req.fun = 1  # Use the constant for setting digital output
        self.req.pin = 16  # Specify the pin for the tool digital output
        self.req.state = 1.0  # Set the desired state (on/off)
        try:
            self.cli(self.req)
            time.sleep(0.5)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")


def main():
    io_client = ActiveGripperControl()
    io_client.open()
    io_client.grasp()
    io_client.grasp()


if __name__ == '__main__':
    main()
