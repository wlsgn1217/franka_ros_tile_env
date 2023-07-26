import rospy
import numpy as np
from franka_msgs.msg import FrankaState
import json

data = {"pose": [], "force": []}


def franka_state_subscriber_callback(msg):
    O_T_EE = msg.O_T_EE
    force = msg.O_F_ext_hat_K
    pose = np.array(O_T_EE).reshape(4,4).transpose(1,0)

    data["pose"] = data["pose"] + pose
    
   
def record(data):
    pass



if __name__ == "__main__":
    rospy.init_node("recorder")
    rate = rospy.Rate(200)

    franka_state_subscriber = rospy.Subscriber("franka_state_controller/franka_states", FrankaState, franka_state_subscriber_callback)

    rospy.spin()