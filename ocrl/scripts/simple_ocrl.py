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
from angles import *

import tf

class SimpleOcrlNode:
  """base class for processing waypoints to give control output"""
  def __init__(self):
    # Parameters
    self.nominal_speed = 3
    #
    self.cmd_pub = rospy.Publisher('/ackermann_vehicle/ackermann_cmd', AckermannDriveStamped, queue_size=10)
    self.spline_path_pub = rospy.Publisher('/spline_path', Path, queue_size=10)
    self.track_point_pub = rospy.Publisher('/track_point', PoseStamped, queue_size=10)
    

    # waypoints = np.zeros((num_waypoints, 3))
    rospy.Subscriber("/ackermann_vehicle/waypoints",
                     PoseArray,
                     self.waypointCallback)
    self.waypoints = np.zeros((num_waypoints, 3))
    self.got_waypoints = False
    rospy.wait_for_message("/ackermann_vehicle/waypoints", PoseArray, 5)
    


    self.rear_axle_center = Pose()
    self.rear_axle_velocity = Twist()
    self.rear_axle_theta = 0
    rospy.Subscriber("/ackermann_vehicle/ground_truth/state",
                     Odometry, self.vehicleStateCallback)
    # rospy.wait_for_message("/ackermann_vehicle/ground_truth/state", Odometry, 5)

    rospy.wait_for_message("/spline_path", Path, 5)
    self.spline_start_time = rospy.Time.now()
    self.track_pt_timer = rospy.Timer(rospy.Duration(0.02), self.trackPointTimerCallback) # track point based on time from spline_path start time
    # rospy.Subscriber("/track_point", PoseStamped, self.trackPointCallback)
    # ## RVIZ Publisher
    # for w in self.waypoints:
    #   self.pursuitToWaypoint(w)
    # traj_controller(self)
    # 
    # self.traj_controller()


  # Keep this from pure_pursuit.py
  def waypointCallback(self,msg):
    # global waypoints
    
    for i in range(len(msg.poses)):
      self.waypoints[i, 0] = msg.poses[i].position.x
      self.waypoints[i, 1] = msg.poses[i].position.y
      self.waypoints[i, 2] = euler_from_quaternion([msg.poses[i].orientation.x, msg.poses[i].orientation.y, msg.poses[i].orientation.z, msg.poses[i].orientation.w])[2]
    self.got_waypoints = True
    self.fitSpline(self.waypoints)

  # Keep this from pure_pursuit.py
  def vehicleStateCallback(self,msg):
    # global rear_axle_center, rear_axle_theta, rear_axle_velocity
    self.rear_axle_center.position.x = msg.pose.pose.position.x
    self.rear_axle_center.position.y = msg.pose.pose.position.y
    self.rear_axle_center.orientation = msg.pose.pose.orientation

    self.rear_axle_theta = euler_from_quaternion(
      [self.rear_axle_center.orientation.x, self.rear_axle_center.orientation.y, self.rear_axle_center.orientation.z,
      self.rear_axle_center.orientation.w])[2]

    self.rear_axle_velocity.linear = msg.twist.twist.linear
    self.rear_axle_velocity.angular = msg.twist.twist.angular


  def fitSpline(self,waypoints): 
      
      waypoints = np.insert(waypoints, 0, [0,0,0], axis=0)# prepend zero state to waypoints # TODO: check yaw
      # Linear length along the line:
      # (https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python)
      distance = np.cumsum( np.sqrt(np.sum( np.diff(waypoints[:,:2], axis=0)**2, axis=1 )) )
      distance = np.insert(distance, 0, 0)/distance[-1]

      # Interpolate 
      interpolation_method = 'cubic' # (quadratic or cubic method worked best)
      alpha = np.linspace(0,1,500) # discretize
      interp_curve =  interp1d(distance, waypoints, kind=interpolation_method, axis=0)
      interp_points = interp_curve(alpha)

      # Publish as nav_msgs/Path message 
      path_msg = Path()
      path_msg.header.stamp = rospy.Time.now()
      path_msg.header.frame_id = '/map'
      for i in range(len(alpha)):
        pose = PoseStamped() 
        pose.pose.position.x = interp_points[i,0]
        pose.pose.position.y = interp_points[i,1]
        path_msg.poses.append(pose)
      
      
      self.spline_path_pub.publish(path_msg)
      self.spline_points = interp_points
      self.spline_distance = np.sum(np.sqrt(np.sum(np.diff(interp_points[:,:2], axis=0)**2, axis=1)))
      self.spline_cum_dist = np.cumsum(np.sqrt(np.sum(np.diff(interp_points[:,:2], axis=0)**2, axis=1)))
      print("Published Spline Path. Distance (m): ", self.spline_distance)

  def trackPointTimerCallback(self, event):
    print("trackPointTimerCallback")
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

    # Copy from pure pursuit.py
    # dx = track_point_x - self.rear_axle_center.position.x
    # dy = track_point_y - self.rear_axle_center.position.y
    # target_distance = math.sqrt(dx*dx + dy*dy)

    
    # cmd.header.stamp = rospy.Time.now()
    # cmd.header.frame_id = "base_link"
    # cmd.drive.speed = self.rear_axle_velocity.linear.x
    # cmd.drive.acceleration = max_acc
    # while target_distance > waypoint_tol:

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

    target_distance = math.sqrt(dx * dx + dy * dy)

    self.cmd_pub.publish(cmd) # CMD includes steering_angle

  # Replace this with our function. They take a pure pursuit approach, 
  # using waypoint as lookahead point, then calculating angle to lookahead point
  # while keeping speed steady
  def pursuitToWaypoint(self, waypoint):
    print(waypoint)
    # global rear_axle_center, rear_axle_theta, rear_axle_velocity, cmd_pub
    rospy.wait_for_message("/ackermann_vehicle/ground_truth/state", Odometry, 5)
    dx = waypoint[0] - self.rear_axle_center.position.x
    dy = waypoint[1] - self.rear_axle_center.position.y
    target_distance = math.sqrt(dx*dx + dy*dy)

    cmd = AckermannDriveStamped()
    cmd.header.stamp = rospy.Time.now()
    cmd.header.frame_id = "base_link"
    cmd.drive.speed = self.rear_axle_velocity.linear.x
    cmd.drive.acceleration = max_acc
    while target_distance > waypoint_tol:

      dx = waypoint[0] - self.rear_axle_center.position.x
      dy = waypoint[1] - self.rear_axle_center.position.y
      lookahead_dist = np.sqrt(dx * dx + dy * dy)
      lookahead_theta = math.atan2(dy, dx)
      alpha = shortest_angular_distance(self.rear_axle_theta, lookahead_theta)

      cmd.header.stamp = rospy.Time.now()
      cmd.header.frame_id = "base_link"
      # Publishing constant speed of 1m/s
      cmd.drive.speed = 1

      # Reactive steering
      if alpha < 0:
        st_ang = max(-max_steering_angle, alpha)
      else:
        st_ang = min(max_steering_angle, alpha)

      cmd.drive.steering_angle = st_ang

      target_distance = math.sqrt(dx * dx + dy * dy)

      self.cmd_pub.publish(cmd) # CMD includes steering_angle
      rospy.wait_for_message("/ackermann_vehicle/ground_truth/state", Odometry, 5)


if __name__ == '__main__':

  rospy.init_node('simple_ocrl_node')
  rospy.loginfo('simple_ocrl_node initialized')
  node = SimpleOcrlNode()
  rospy.spin()




