import rospy
from fs_msgs.msg import ControlCommand
from std_msgs.msg import Int32
import numpy as np
from nav_msgs.msg import Odometry

class OdometerToVel():
    def __init__(self):
        self.steering = 0
        self.wheel = 0.2
        self.wheel_circ = 2 * np.pi * self.wheel
        self.initializePublishers()
        
    def initializeSubscribers(self):
        rospy.Subscriber('/control_command', ControlCommand, self.getControlCommand)
        rospy.Subscriber('/rc_odometer', Int32, self.getRPS)
        rospy.INFO('Subscribed to topics')

    def initializePublishers(self):
        self.pub = rospy.Publisher('/wheel_speed', Odometry, queue_size=1)
        rospy.INFO('Publisher initialized')

    def getControlCommand(self, data):
        self.steering = data.steering
    
    def getRPS(self, data):
        rps = data.data
        speed = self.wheel_circ * rps
        angle = ((180) / (-1 - 1)) * (self.steering - 1)
        vx = speed * np.sin(angle)
        vy = speed * np.cos(angle)
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = 'odometer'
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        self.pub.publish(odom)