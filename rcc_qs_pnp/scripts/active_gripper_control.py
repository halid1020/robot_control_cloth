#!/usr/bin/env python3

from ur_msgs.srv import SetIO
import rclpy
from rclpy.node import Node
import time


class ActiveGripperControl(Node):
    def __init__(self):
        super().__init__('client_test')
        self.cli = self.create_client(SetIO, '/io_and_status_controller/set_io')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = SetIO.Request()

    def open(self):
        self.req.fun = 1  # Use the constant for setting digital output
        self.req.pin = 16     # Specify the pin for the tool digital output
        self.req.state = 0.0           # Set the desired state (on/off)
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        time.sleep(1)
    
    def grasp(self):
        self.req.fun = 1  # Use the constant for setting digital output
        self.req.pin = 16     # Specify the pin for the tool digital output
        self.req.state = 1.0           # Set the desired state (on/off)
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        time.sleep(1)

def main():
    #print('pin id', SetIO.PIN_TOOL_DOUT1 )
    rclpy.init()
    io_client = ActiveGripperControl()
    io_client.open()
    io_client.grasp()
    io_client.grasp()
    rclpy.shutdown()

if __name__ == '__main__':
    main()