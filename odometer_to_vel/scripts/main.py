import rospy
from odometer_to_vel import OdometerToVel

def main():
    rospy.init_node('odometer_to_vel')
    odom = OdometerToVel()
    odom.initializeSubscribers()
    
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass