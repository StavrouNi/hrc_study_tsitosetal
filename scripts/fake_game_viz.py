#!/usr/bin/env python
import tf
import rospy
from cartesian_state_msgs.msg import PoseTwist
from visualization_msgs.msg import MarkerArray, Marker

class FakeGameViz:
    def __init__(self):
        node_of_interest = rospy.get_param("~node_of_interest", "robot_movement_generation")
        s0 = rospy.get_param("/"+node_of_interest+"/start_position_0", [0.0,0.0,0.0])
        s1 = rospy.get_param("/"+node_of_interest+"/start_position_1", [0.0,0.0,0.0])
        s2 = rospy.get_param("/"+node_of_interest+"/start_position_2", [0.0,0.0,0.0])
        s3 = rospy.get_param("/"+node_of_interest+"/start_position_3", [0.0,0.0,0.0])
        goal = [(s0[0]+s1[0]+s2[0]+s3[0])/4.0,(s0[1]+s1[1]+s2[1]+s3[1])/4.0,(s0[2]+s1[2]+s2[2]+s3[2])/4.0]
        input_topic = rospy.get_param("~input_topic", "/ur3_cartesian_velocity_controller/ee_state")
        output_topic = rospy.get_param("~output_topic", "/fake_game_viz")
        self.marker_frame_id = rospy.get_param("~marker_frame_id", "world")

        self.markers = MarkerArray()
        self.markers.markers.append(self.create_marker(110,1,0,goal[0],goal[1],0,0.01,0.01,0.01,0.0,1.0,0.0))
        self.markers.markers.append(self.create_marker(111,1,0,s0[0],s0[1],0,0.01,0.01,0.01,1.0,0.0,0.0))
        self.markers.markers.append(self.create_marker(112,1,0,s1[0],s1[1],0,0.01,0.01,0.01,1.0,0.0,0.0))
        self.markers.markers.append(self.create_marker(113,1,0,s2[0],s2[1],0,0.01,0.01,0.01,1.0,0.0,0.0))
        self.markers.markers.append(self.create_marker(114,1,0,s3[0],s3[1],0,0.01,0.01,0.01,1.0,0.0,0.0))
        # This is used for the end effector
        self.markers.markers.append(self.create_marker(115,3,0,0,0,0,0.05,0.05,0.05,0.0,0.0,1.0))
        # This is used for the projection of the end effector
        self.markers.markers.append(self.create_marker(116,3,0,0,0,0,0.01,0.01,0.01,1.0,0.0,1.0))

        self.sub = rospy.Subscriber(input_topic, PoseTwist, self.state_callback)
        self.pub = rospy.Publisher(output_topic, MarkerArray, queue_size=1)

    def create_marker(self, mid, tid, aid, x, y, z, sx, sy, sz, r, g, b):
        marker = Marker()
        marker.header.frame_id = self.marker_frame_id
        marker.id = mid
        marker.type = tid
        marker.action = aid
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1
        marker.scale.x = sx
        marker.scale.y = sy
        marker.scale.z = sz
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0
        return marker

    def state_callback(self, msg):
        self.markers.markers[-2].pose.position.x = msg.pose.position.x
        self.markers.markers[-2].pose.position.y = msg.pose.position.y
        self.markers.markers[-2].pose.position.z = msg.pose.position.z
        self.markers.markers[-1].pose.position.x = msg.pose.position.x
        self.markers.markers[-1].pose.position.y = msg.pose.position.y
        self.pub.publish(self.markers)

if __name__ == "__main__":
    rospy.init_node("fake_game_viz")
    FakeGameViz()
    rospy.spin()

