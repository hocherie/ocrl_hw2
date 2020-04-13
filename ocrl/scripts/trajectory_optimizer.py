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
from scipy.optimize import minimize
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
    
    # Initialize Subscribers and relevant variables
    rospy.Subscriber("/ackermann_vehicle/waypoints",
                     PoseArray,
                     self.waypointCallback) # also outputs spline path
    self.waypoints = np.zeros((num_waypoints, 3))
    self.got_waypoints = False
    rospy.wait_for_message("/ackermann_vehicle/waypoints", PoseArray, 5)
   

  # Keep this from pure_pursuit.py
  def waypointCallback(self,msg):
    if self.got_waypoints == False:
      for i in range(len(msg.poses)):
        self.waypoints[i, 0] = msg.poses[i].position.x
        self.waypoints[i, 1] = msg.poses[i].position.y
        self.waypoints[i, 2] = euler_from_quaternion([msg.poses[i].orientation.x, msg.poses[i].orientation.y, msg.poses[i].orientation.z, msg.poses[i].orientation.w])[2]
      self.fitSpline(self.waypoints)


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
      
  def optimize_path(self,segments,path):
    N = len(path)
    acc = np.ones((1,N)).flatten() 
    steer = np.zeros((1,N)).flatten()
    idx = {'x':0, 'y':1, 'o':2, 'v':3, 't':4, 'acc':5, 'steer':6}


    x0 = path.flatten('F')
    x0 = np.hstack((x0,acc,steer))


    
    for variable,i in idx.items():
      pass
      #print("{0}: {1}".format(variable,x0[N*i:N*(i+1)]))

    def objective(x,seg=segments):
      final_time = x[N*(idx['t']+1)-1]
      cost = final_time
      
      print cost
      return cost
    def waypoint_constraint(x,seg = segments):
      n_segments = len(segments)
      len_segments = [len(segments[i]) for i in range(n_segments)]
      x_diffs = np.ones((n_segments,1))
      y_diffs = np.ones((n_segments,1))
      o_diffs = np.ones((n_segments,1))

      for seg_i,length in enumerate(len_segments):
        x_diffs[seg_i] = x[idx['x']+length-1] - segments[seg_i][0][0]
        y_diffs[seg_i] = x[idx['y']+length-1] - segments[seg_i][0][1]
        o_diffs[seg_i] = x[idx['o']+length-1] - segments[seg_i][0][2]
      
      x_err = np.sum(x_diffs[:])
      y_err = np.sum(y_diffs[:])
      o_err = np.sum(o_diffs[:])
      #print("x:{},y:{},o:{}".format(x_err,y_err,o_err))
      return np.sum([x_err,y_err,o_err]) 
    def dynamics_constraint(x):
      return 0
    def time_constraint(x):
      times = x[N*idx['t']:x[N*(idx['t']+1)]]
      return np.sum(np.diff(times))
    
    c1 = {'type':'eq', 'fun' : waypoint_constraint}
    c2 = {'type':'eq', 'fun' : dynamics_constraint}
    c3 = {'type':'ineq', 'fun' : time_constraint}
    cons = [c1,c3]

    onev = np.ones((N,1))
    xub = 15*onev
    xlb = -15*onev
    yub = 15*onev
    ylb = -15*onev
    oub = np.inf*onev 
    olb = -np.inf*onev
    vub = 10*onev
    vlb = -10*onev
    tub = x0[N*(idx['t']+1)-1]*onev
    tlb = 0*onev
    aub = 4*onev
    alb = -4*onev
    sub = np.pi/6*onev
    slb = -np.pi/6*onev

    ub = np.vstack((xub, yub, oub, vub, tub, aub, sub))
    lb = np.vstack((xlb, ylb, olb, vlb, tlb, alb, slb))
    if (lb > ub).any():
      return None
    bnds = np.hstack((ub,lb))
    print(bnds.shape)
    
    sol= minimize(objective,x0, constraints=cons)
    print(sol)
    optimal_path = sol.x.reshape((N,7))
    return optimal_path
  def fitSpline(self,waypoints): 
      # spline configurations
      turning_radius = 0.67
      step_size = 0.5

      waypoints = np.insert(waypoints, 0, [0,0,0], axis=0)# prepend zero state to waypoints # TODO: check yaw

      # find heading-fitting spline
      path_list = np.empty((0,3))
      segments = []
      for i in range(waypoints.shape[0] - 1):
          q0 = (waypoints[i,0], waypoints[i,1], waypoints[i,2])
          q1 = (waypoints[i+1,0], waypoints[i+1,1], waypoints[i+1,2])

          path = dubins.shortest_path(q0, q1, turning_radius)
          configurations, _ = path.sample_many(step_size)#first element of configurations is the waypoint before that segment
          configurations = np.array(configurations)
          segments.append(configurations)
          path_list = np.vstack((path_list, configurations))
      self.spline_points = path_list
      self.spline_distance = np.sum(np.sqrt(np.sum(np.diff(path_list[:,:2], axis=0)**2, axis=1)))
      

      dx = np.gradient(path_list[:,0])
      dy = np.gradient(path_list[:,1])
      vel_vec = np.vstack((dx,dy))
      velocities = np.linalg.norm(vel_vec,axis=0).reshape(-1,1)

      total_time = self.spline_distance/self.nominal_speed
      print("Duration of trajectory={}".format(total_time))
      self.dt = total_time/len(path_list)
      times =  np.array([(i*self.dt) for i in range(path_list.shape[0])])
      path_list = np.hstack((path_list,velocities,times.reshape(-1,1)))
      print(path_list)
      optimized_path = self.optimize_path(segments,path_list)
      print(optimized_path)
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
      print("Published Spline Path. Distance (m): ", self.spline_distance)

if __name__ == '__main__':

  rospy.init_node('trajectory_optimizer_node')
  rospy.loginfo('trajectory_optimizer_node initialized')
  node = TrajectoryOptimizerNode()
  rospy.spin()




