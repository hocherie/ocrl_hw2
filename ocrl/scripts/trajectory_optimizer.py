#!/usr/bin/env python
"""
OCRL HW2
Simple: first fit a spline for received waypoints, then a path tracking or PID controller to follow
"""

from common import *
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseArray, Pose, Twist, PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
# from visualization_msgs
from scipy.interpolate import interp1d
import dubins
from angles import *

import tf

class TrajectoryOptimizerNode:
  """base class for processing waypoints to give control output"""
  def __init__(self):
    # Parameters
    self.nominal_speed = 2 

    # Initialize Publishers
    self.cmd_pub = rospy.Publisher('/ackermann_vehicle/ackermann_cmd', AckermannDriveStamped, queue_size=10)
    self.spline_path_pub = rospy.Publisher('/spline_path', Path, queue_size=10)
    self.track_point_pub = rospy.Publisher('/track_point', PoseStamped, queue_size=10)
    # self.completed_waypoint_pub = rospy.Publisher('/completed_waypoints', )
    
    # Initialize Subscribers and relevant variables
    rospy.Subscriber("/ackermann_vehicle/waypoints",
                     PoseArray,
                     self.waypointCallback) # also outputs spline path
    self.waypoints = np.zeros((num_waypoints, 3))
    self.got_waypoints = False
    rospy.wait_for_message("/ackermann_vehicle/waypoints", PoseArray, 5)
    self.got_vehicle_state = False
    self.rear_axle_center = Pose()
    self.rear_axle_velocity = Twist()
    self.rear_axle_theta = 0
    rospy.Subscriber("/ackermann_vehicle/ground_truth/state",
                     Odometry, self.vehicleStateCallback)

   

  # Keep this from pure_pursuit.py
  def waypointCallback(self,msg):
    if self.got_waypoints == False:
      for i in range(len(msg.poses)):
        self.waypoints[i, 0] = msg.poses[i].position.x
        self.waypoints[i, 1] = msg.poses[i].position.y
        self.waypoints[i, 2] = euler_from_quaternion([msg.poses[i].orientation.x, msg.poses[i].orientation.y, msg.poses[i].orientation.z, msg.poses[i].orientation.w])[2]
      self.fitSpline(self.waypoints)

  # Keep this from pure_pursuit.py
  def vehicleStateCallback(self,msg):
    if self.got_vehicle_state == False:
      self.rear_axle_center.position.x = msg.pose.pose.position.x
      self.rear_axle_center.position.y = msg.pose.pose.position.y
      self.rear_axle_center.orientation = msg.pose.pose.orientation

      self.rear_axle_theta = euler_from_quaternion(
        [self.rear_axle_center.orientation.x, self.rear_axle_center.orientation.y, self.rear_axle_center.orientation.z,
        self.rear_axle_center.orientation.w])[2]

      self.rear_axle_velocity.linear = msg.twist.twist.linear
      self.rear_axle_velocity.angular = msg.twist.twist.angular
      self.got_vehicle_state = True

  @staticmethod
  def theta_from_path(path, index, mode = 'forward'):
      current = path[index]
      if index != 0:
        prev = index-1
      if index != len(path)-1:
        succ = index+1
      
      if mode == 'forward':
        vector = np.array([1,0])
      theta = np.arctan2(vector[1],vector[0])
      return theta
      
  def fitSpline(self,waypoints): 
      # spline configurations
      turning_radius = 2.5
      step_size = 0.1

      waypoints = np.insert(waypoints, 0, [0,0,0], axis=0)# prepend zero state to waypoints # TODO: check yaw

      # find heading-fitting spline
      path_list = np.empty((0,3))
      for i in range(waypoints.shape[0] - 1):
          q0 = (waypoints[i,0], waypoints[i,1], waypoints[i,2])
          q1 = (waypoints[i+1,0], waypoints[i+1,1], waypoints[i+1,2])

          path = dubins.shortest_path(q0, q1, turning_radius)
          configurations, _ = path.sample_many(step_size)
          configurations = np.array(configurations)
          path_list = np.vstack((path_list, configurations))

      self.spline_points = path_list
      self.spline_distance = np.sum(np.sqrt(np.sum(np.diff(path_list[:,:2], axis=0)**2, axis=1)))
      self.spline_cum_dist = np.cumsum(np.sqrt(np.sum(np.diff(path_list[:,:2], axis=0)**2, axis=1)))
      
      total_time = self.spline_distance/self.nominal_speed
      #print("Duration of trajectory={}".format(total_time))
      self.dt = total_time/len(path_list)
      # Publish as nav_msgs/Path message 
      path_msg = Path()
      path_msg.header.stamp = rospy.Time.now()
      path_msg.header.frame_id = '/map'
      for i in range(path_list.shape[0]):
        pose = PoseStamped() 
        pose.pose.position.x = path_list[i,0]
        pose.pose.position.y = path_list[i,1]
        theta = path_list[i,2]
        quat = quaternion_from_euler(0,0,theta)
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        pose.header.stamp = rospy.Time.from_sec(i*self.dt)
        path_msg.poses.append(pose)
  
      

      self.spline_path_pub.publish(path_msg)
      #print("Published Spline Path. Distance (m): ", self.spline_distance)

if __name__ == '__main__':

  rospy.init_node('trajectory_optimizer_node')
  rospy.loginfo('trajectory_optimizer_node initialized')
  node = TrajectoryOptimizerNode()
  rospy.spin()




