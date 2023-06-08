#! /usr/bin/env python
import time
import rospy

from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelConfiguration

def set_model_configuration_client(model_name, model_param_name, joint_names, joint_positions, gazebo_namespace):
    rospy.wait_for_service(gazebo_namespace+'/set_model_configuration')
    time.sleep(2)
    try:
        set_model_configuration = rospy.ServiceProxy(gazebo_namespace+'/set_model_configuration', SetModelConfiguration)
        resp = set_model_configuration(model_name, model_param_name, joint_names, joint_positions)
        unpause = rospy.ServiceProxy(gazebo_namespace+"/unpause_physics", Empty)
        time.sleep(2)
        unpause()
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

rospy.init_node("set_init_position_node")
set_model_configuration_client("robot", "robot_description", ["elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], [-1.8, -2.2, -0.3, -2.2, -1.5, 0.5], "gazebo")
