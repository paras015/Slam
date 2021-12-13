#!/usr/bin/env python3

import rospy
from fastSlam import FastSLAM

def main():
    rospy.init_node('fast_slam')
    fs = FastSLAM()
    fs.initializeSubscribers()
    while not rospy.is_shutdown():
    #     print("AA")
        rospy.spin()
    
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass