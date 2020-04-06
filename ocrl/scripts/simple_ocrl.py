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

class SimpleOcrlNode:
  """base class for processing waypoints to give control output"""
  def __init__(self):
    # Parameters
    self.nominal_speed = 3 

    # Initialize Publishers
    self.cmd_pub = rospy.Publisher('/ackermann_vehicle/ackermann_cmd', AckermannDriveStamped, queue_size=10)
    self.spline_path_pub = rospy.Publisher('/spline_path', Path, queue_size=10)
    self.track_point_pub = rospy.Publisher('/track_point', PoseStamped, queue_size=10)
    
    # Initialize Subscribers and relevant variables
    rospy.Subscriber("/ackermann_vehicle/waypoints",
                     PoseArray,
                     self.waypointCallback) # also outputs spline path
    self.waypoints = np.zeros((num_waypoints, 3))
    self.got_waypoints = False
    rospy.wait_for_message("/ackermann_vehicle/waypoints", PoseArray, 5)
    
    self.rear_axle_center = Pose()
    self.rear_axle_velocity = Twist()
    self.rear_axle_theta = 0
    rospy.Subscriber("/ackermann_vehicle/ground_truth/state",
                     Odometry, self.vehicleStateCallback)

    # Marks time we get first spline path as spline_start_time, and starts outputting tracking point and associated commands
    rospy.wait_for_message("/spline_path", Path, 5)
    self.spline_start_time = rospy.Time.now()
    self.track_pt_timer = rospy.Timer(rospy.Duration(0.02), self.trackPointTimerCallback) # track point based on time from spline_path start time


  # Keep this from pure_pursuit.py
  def waypointCallback(self,msg):
    for i in range(len(msg.poses)):
      self.waypoints[i, 0] = msg.poses[i].position.x
      self.waypoints[i, 1] = msg.poses[i].position.y
      self.waypoints[i, 2] = euler_from_quaternion([msg.poses[i].orientation.x, msg.poses[i].orientation.y, msg.poses[i].orientation.z, msg.poses[i].orientation.w])[2]
    self.got_waypoints = True
    self.fitSpline(self.waypoints)

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


  def fitSpline(self,waypoints): 
      # spline configurations
      turning_radius = 1
      step_size = 0.5

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


      # Publish as nav_msgs/Path message 
      path_msg = Path()
      path_msg.header.stamp = rospy.Time.now()
      path_msg.header.frame_id = '/map'
      for i in range(path_list.shape[0]):
        pose = PoseStamped() 
        pose.pose.position.x = path_list[i,0]
        pose.pose.position.y = path_list[i,1]
        path_msg.poses.append(pose)
      
      
      self.spline_path_pub.publish(path_msg)
      self.spline_points = path_list
      self.spline_distance = np.sum(np.sqrt(np.sum(np.diff(path_list[:,:2], axis=0)**2, axis=1)))
      self.spline_cum_dist = np.cumsum(np.sqrt(np.sum(np.diff(path_list[:,:2], axis=0)**2, axis=1)))
      print("Published Spline Path. Distance (m): ", self.spline_distance)

  def trackPointTimerCallback(self, event):
    time_since_start = (rospy.Time.now() - self.spline_start_time).to_sec() 
    dist_along_spline = self.nominal_speed * time_since_start
    track_point_ind = np.argwhere(self.spline_cum_dist > dist_along_spline)[0]
    track_point_x = self.spline_points[track_point_ind, 0]
    track_point_y = self.spline_points[track_point_ind, 1]
    # Publish track point pose
    track_pose_msg = PoseStamped() 
    track_pose_msg.header.stamp = rospy.Time.now() 
    track_pose_msg.header.frame_id = '/map'
    track_pose_msg.pose.position.x = track_point_x
    track_pose_msg.pose.position.y = track_point_y
    self.track_point_pub.publish(track_pose_msg)

    # Calculate Commands based on Tracking Point
    dx = track_point_x - self.rear_axle_center.position.x
    dy = track_point_y - self.rear_axle_center.position.y
    lookahead_dist = np.sqrt(dx * dx + dy * dy)
    lookahead_theta = math.atan2(dy, dx)
    alpha = shortest_angular_distance(self.rear_axle_theta, lookahead_theta)

    cmd = AckermannDriveStamped()
    cmd.header.stamp = rospy.Time.now()
    cmd.header.frame_id = "base_link"
    # Publishing constant speed of 1m/s
    cmd.drive.speed = self.nominal_speed

    # Reactive steering
    if alpha < 0:
      st_ang = max(-max_steering_angle, alpha)
    else:
      st_ang = min(max_steering_angle, alpha)

    cmd.drive.steering_angle = st_ang

    self.cmd_pub.publish(cmd) # CMD includes steering_angle


if __name__ == '__main__':

  rospy.init_node('simple_ocrl_node')
  rospy.loginfo('simple_ocrl_node initialized')
  node = SimpleOcrlNode()
  rospy.spin()




