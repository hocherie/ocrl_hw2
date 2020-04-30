#!/usr/bin/env python
"""
OCRL HW2
Simple: first fit a spline for received waypoints, then a path tracking or PID controller to follow
"""

from common import *
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseArray, Pose, Twist, PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from scipy.interpolate import interp1d
import dubins
from angles import *

import tf

import numpy as np
from lqr_functions import *
from spline_functions import *

class LqrNode:
  """base class for processing waypoints to give control output"""
  def __init__(self,max_speed,q_gain,r_gain,turning_radius,f):
    self.trajectory_file = f
    self.last_pose_string = ""
    self.num_repeats = 0
    self.waypoints_hit = set()
    # Parameters
    # self.target_speed = 10.0
    self.target_speed = max_speed
    self.lqr_params = dict()
    self.lqr_params['maxsimtime'] = 20.0
    self.lqr_params['goal_dis'] = 0.3
    self.lqr_params['stop_speed'] = 0.05
    self.lqr_params['lqr_Q'] = q_gain*np.eye(5)
    self.lqr_params['lqr_R'] = r_gain*np.eye(2)
    self.lqr_params['wheelbase'] = 0.335
    self.lqr_params['max_steer'] = np.deg2rad(20.0)
    self.e, self.e_th = 0.0, 0.0
    self.last_ind = 0
    self.is_not_done = True

    # Initialize Publishers
    self.cmd_pub = rospy.Publisher('/ackermann_vehicle/ackermann_cmd', AckermannDriveStamped, queue_size=10)
    self.track_point_pub = rospy.Publisher('/track_point', PoseStamped, queue_size=10)
    
    # Initialize Subscribers and relevant variables
    
    self.got_spline = False
    rospy.Subscriber("/spline_path",Path,self.trajectoryCallback)

    rospy.Subscriber("/ackermann_vehicle/waypoints",
                     PoseArray,
                     self.waypointCallback)
    self.waypoints = np.zeros((num_waypoints, 3))
    self.got_waypoints = False

    self.rear_axle_center = Pose()
    self.rear_axle_velocity = Twist()
    self.rear_axle_theta = 0
    rospy.Subscriber("/ackermann_vehicle/ground_truth/state",
                     Odometry, self.vehicleStateCallback)

    # Marks time we get first spline path as spline_start_time, and starts outputting tracking point and associated commands
    rospy.wait_for_message("/spline_path", Path, 10)
    self.got_spline = True
    self.spline_start_time = rospy.Time.now()
    self.track_pt_timer = rospy.Timer(rospy.Duration(0.02), self.trackPointTimerCallback) # track point based on time from spline_path start tim

  def waypointCallback(self,msg):
    if self.got_waypoints == False:
      for i in range(len(msg.poses)):
        self.waypoints[i, 0] = msg.poses[i].position.x
        self.waypoints[i, 1] = msg.poses[i].position.y
        self.waypoints[i, 2] = euler_from_quaternion([msg.poses[i].orientation.x, msg.poses[i].orientation.y, msg.poses[i].orientation.z, msg.poses[i].orientation.w])[2]
        self.waypoints = np.array(self.waypoints)
      self.got_waypoints = True
        
  def trajectoryCallback(self,msg):
    path_list = []

    for i in range(len(msg.poses)):
      pose_i = msg.poses[i].pose
      theta = euler_from_quaternion([pose_i.orientation.x, pose_i.orientation.y, pose_i.orientation.z, pose_i.orientation.w])[2]
      path_list.append([pose_i.position.x, pose_i.position.y, theta, msg.poses[i].header.stamp.secs])
  
    path_list = np.array(path_list)
    # print(path_list)
    self.spline_points = path_list
    self.spline_distance = np.sum(np.sqrt(np.sum(np.diff(path_list[:,:2], axis=0)**2, axis=1)))
    self.spline_cum_dist = np.cumsum(np.sqrt(np.sum(np.diff(path_list[:,:2], axis=0)**2, axis=1)))

  # Keep this from pure_pursuit.py
  def vehicleStateCallback(self,msg):

    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    theta = euler_from_quaternion([msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w])[2]
    self.rear_axle_center.position.x = x
    self.rear_axle_center.position.y = y
    self.rear_axle_theta = theta
    self.rear_axle_velocity.linear = msg.twist.twist.linear
    self.rear_axle_velocity.angular = msg.twist.twist.angular
    
    pose_string = "{:0.2f},{:0.2f},{:0.3f}\n".format(self.rear_axle_center.position.x , self.rear_axle_center.position.y, theta)

    cur_pos_2d = np.array([x,y])
    position_error = np.round(np.abs(np.linalg.norm(self.waypoints[:,:2]-cur_pos_2d,axis=1)),decimals=2)
    angle_error = np.round(np.abs(np.degrees(self.waypoints[:,2]-theta)),decimals=3)
    waypoint_tol = 0.2 
    waypoint_ang_tol = 5
    pos_correct = position_error < waypoint_tol
    ang_correct = angle_error < waypoint_ang_tol
    both_correct = np.bitwise_and(ang_correct,pos_correct)
    if both_correct.any():
      way_idx = np.nonzero(both_correct)
      wi = way_idx[0][0]
      if wi not in self.waypoints_hit:
        self.waypoints_hit.add(wi)
        print("Got within <{} m,{} degrees> of waypoint {}.".format(position_error[wi], angle_error[wi],wi+1))
        
    self.last_pose_string = pose_string
    

   

    # Stop when end of waypoints

    
    if self.got_waypoints:
      dist_to_goal = np.linalg.norm(self.spline_points[-1,0:2] - cur_pos_2d)
      ang_to_goal = np.linalg.norm(self.waypoints[-1,2]-theta)
      if (dist_to_goal < waypoint_tol and ang_to_goal < waypoint_ang_tol) and self.is_not_done:
        self.trajectory_file.write("{}".format(list(self.waypoints_hit)))
        print("Done!!")
        self.is_not_done = False

# ---------------------------------------------



  def trackPointTimerCallback(self, event):
    if self.is_not_done:

      cmd = AckermannDriveStamped()
      cmd.header.stamp = rospy.Time.now()
      cmd.header.frame_id = "base_link"


      # -------------------

      self.lqr_params['dt'] = 0.02 #self.spline_points[-1,3] - self.spline_points[-2,3]
      # print("DT", self.spline_points)

      goal = [self.spline_points[-1,0], self.spline_points[-1,1]] # goal is last x, y point in spline
      cx = self.spline_points[:,0]
      cy = self.spline_points[:,1]
      cyaw = self.spline_points[:,2]
      ck = curvature(cx, cy, self.lqr_params['dt'])

      speed_profile = calc_speed_profile(cyaw, self.target_speed)

      # ------------------------------

      T = self.lqr_params['maxsimtime']
      goal_dis = self.lqr_params['goal_dis']
      stop_speed = self.lqr_params['stop_speed']
      lqr_Q = self.lqr_params['lqr_Q']
      lqr_R = self.lqr_params['lqr_R']
      dt = self.lqr_params['dt']
      wheelbase = self.lqr_params['wheelbase']

      x = self.rear_axle_center.position.x
      y = self.rear_axle_center.position.y
      yaw = self.rear_axle_theta


      # self.rear_axle_velocity.linear
      v = np.linalg.norm([self.rear_axle_velocity.linear.x, self.rear_axle_velocity.linear.y])

      state = State(x=x, y=y, yaw=yaw, v=v)

      # initialize e and e_th above

      dl, target_ind, self.e, self.e_th, ai = lqr_speed_steering_control(
          state, cx, cy, cyaw, ck, self.e, self.e_th, speed_profile, lqr_Q, lqr_R, dt, wheelbase, self.last_ind)
      # print("Target_ind", target_ind)
      # Publish track point pose
      track_pose_msg = PoseStamped() 
      track_pose_msg.header.stamp = rospy.Time.now() 
      track_pose_msg.header.frame_id = '/map'
      track_pose_msg.pose.position.x = cx[target_ind]
      track_pose_msg.pose.position.y = cy[target_ind]
      quat = quaternion_from_euler(0,0,cyaw[target_ind])
      track_pose_msg.pose.orientation.x = quat[0]
      track_pose_msg.pose.orientation.y = quat[1]
      track_pose_msg.pose.orientation.z = quat[2]
      track_pose_msg.pose.orientation.w = quat[3]

      self.track_point_pub.publish(track_pose_msg)



      self.last_ind = target_ind
      cmd.drive.speed = v + ai * dt
      cmd.drive.steering_angle = dl

  

      self.cmd_pub.publish(cmd) # CMD includes steering_angle


if __name__ == '__main__':

  rospy.init_node('lqr_node')
  rospy.loginfo('lqr_node initialized')
  

  max_speed = 3
  q_gain = 1
  r_gain = 1
  turning_radius = 2.5
  with open("speed{}_Q{}_R{}_rad{}.txt".format(max_speed,q_gain,r_gain,turning_radius),'w') as f: 
    node = LqrNode(max_speed,q_gain,r_gain,turning_radius,f)
    rospy.spin()




