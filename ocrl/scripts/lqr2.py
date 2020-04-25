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
  def __init__(self):
    # Parameters
    # self.target_speed = 10.0
    self.target_speed = 3
    self.lqr_params = dict()
    self.lqr_params['maxsimtime'] = 20.0
    self.lqr_params['goal_dis'] = 0.3
    self.lqr_params['stop_speed'] = 0.05
    self.lqr_params['lqr_Q'] = np.eye(5)
    self.lqr_params['lqr_R'] = np.eye(2)
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

  def trajectoryCallback(self,msg):
      print("trajectory callback")
      path_list = []

      for i in range(len(msg.poses)):
        pose_i = msg.poses[i].pose
        theta = euler_from_quaternion([pose_i.orientation.x, pose_i.orientation.y, pose_i.orientation.z, pose_i.orientation.w])[2]
        path_list.append([pose_i.position.x, pose_i.position.y, theta, msg.poses[i].header.stamp.secs])
        #pose.pose.position.x = path_list[i,0]
        #pose.pose.position.y = path_list[i,1]
    
      path_list = np.array(path_list)
      # print(path_list)
      self.spline_points = path_list
      self.spline_distance = np.sum(np.sqrt(np.sum(np.diff(path_list[:,:2], axis=0)**2, axis=1)))
      self.spline_cum_dist = np.cumsum(np.sqrt(np.sum(np.diff(path_list[:,:2], axis=0)**2, axis=1)))

  # Keep this from pure_pursuit.py
  def vehicleStateCallback(self,msg):
    self.rear_axle_center.position.x = msg.pose.pose.position.x
    self.rear_axle_center.position.y = msg.pose.pose.position.y
    self.rear_axle_center.orientation = msg.pose.pose.orientation

    self.rear_axle_theta = euler_from_quaternion(
      [self.rear_axle_center.orientation.x, self.rear_axle_center.orientation.y, self.rear_axle_center.orientation.z,
      self.rear_axle_center.orientation.w])[2]

    self.rear_axle_velocity.linear = msg.twist.twist.linear
    self.rear_axle_velocity.angular = msg.twist.twist.angular

    # Stop when end of waypoints
    waypoint_tol = 0.2 
    cur_pos_2d = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
    
    dist_to_goal = np.linalg.norm(self.spline_points[-1,0:2] - cur_pos_2d)
    print(dist_to_goal)
    if (dist_to_goal < waypoint_tol):
      print("Done!!")
      self.is_not_done = False

# ---------------------------------------------



  def trackPointTimerCallback(self, event):
    if self.is_not_done:
      # print("track point timer callback")
      # time_since_start = (rospy.Time.now() - self.spline_start_time).to_sec() 
      # dist_along_spline = self.nominal_speed * time_since_start
      # track_point_ind = np.argwhere(self.spline_cum_dist > dist_along_spline)[0]
      # track_point_x = self.spline_points[track_point_ind, 0]
      # track_point_y = self.spline_points[track_point_ind, 1]

      # # Publish track point pose
      # track_pose_msg = PoseStamped() 
      # track_pose_msg.header.stamp = rospy.Time.now() 
      # track_pose_msg.header.frame_id = '/map'
      # track_pose_msg.pose.position.x = track_point_x
      # track_pose_msg.pose.position.y = track_point_y
      # self.track_point_pub.publish(track_pose_msg)

      # # Calculate Commands based on Tracking Point
      # dx = track_point_x - self.rear_axle_center.position.x
      # dy = track_point_y - self.rear_axle_center.position.y
      # lookahead_dist = np.sqrt(dx * dx + dy * dy)
      # lookahead_theta = math.atan2(dy, dx)
      # alpha = shortest_angular_distance(self.rear_axle_theta, lookahead_theta)

      cmd = AckermannDriveStamped()
      cmd.header.stamp = rospy.Time.now()
      cmd.header.frame_id = "base_link"


      # # Publishing constant speed of 1m/s
      # cmd.drive.speed = self.target_speed

      # # Reactive steering
      # if alpha < 0:
      #   st_ang = max(-max_steering_angle, alpha)
      # else:
      #   st_ang = min(max_steering_angle, alpha)
      # cmd.drive.steering_angle = st_ang


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

      # TODO is this an issue computing velocity this way?
      # TODO we really need speed? will we never get negative values here?

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

      # TODO
      # IS TARGET SPEED = 10 WHAT WE WANT??????

      
      # TODO calc_nearest_index is not the thing we want to do, it messes with control when spline intersects with itself
      # TODO maybe the curvature of the spline is not being computed right... it is terrible at turning
      # TODO implement stop condition, it just runs around after it 'gets' to the goal


      # -------------------------------

      # t, x, y, yaw, v, delta = do_simulation(cx, cy, cyaw, ck, sp, goal, self.lqr_params)

      # for i in range(1, len(x)):
      #     cmd.drive.speed = v[i]
      #     cmd.drive.steering_angle = delta[i]

      # -------------------------

      self.cmd_pub.publish(cmd) # CMD includes steering_angle


if __name__ == '__main__':

  rospy.init_node('lqr_node')
  rospy.loginfo('lqr_node initialized')
  node = LqrNode()
  rospy.spin()




