## TEST SPLINE FITTING
# Given a set of 2D points, fits spline then plots
import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d


# Vehicle Constants (same as common.py)
x_lim = [-10, 10]
y_lim = [-10, 10]
theta_lim = [-np.pi, np.pi]
num_waypoints = 10
waypoint_tol = 0.2

wheelbase = 0.335
max_acc = 3
max_steering_angle = 0.5

# Generate random waypoints (same as waypoint_publisher.py)
waypoints = np.random.rand(num_waypoints, 3)
waypoints[:, 0] = (x_lim[1] - x_lim[0]) * waypoints[:, 0] + x_lim[0]
waypoints[:, 1] = (y_lim[1] - y_lim[0]) * waypoints[:, 1] + y_lim[0]
waypoints[:, 2] = (theta_lim[1] - theta_lim[0]) * waypoints[:, 2] + theta_lim[0]

# Linear length along the line:
# (https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python)
distance = np.cumsum( np.sqrt(np.sum( np.diff(waypoints[:,:2], axis=0)**2, axis=1 )) )
distance = np.insert(distance, 0, 0)/distance[-1]

# Interpolation for different methods:
# Either quadratic or cubic works best
interpolations_methods = ['slinear', 'quadratic', 'cubic']
alpha = np.linspace(0, 1, 100)

interpolated_points = {}
for method in interpolations_methods:
    interpolator =  interp1d(distance, waypoints, kind=method, axis=0)
    interpolated_points[method] = interpolator(alpha)

# Plot splines
for method_name, curve in interpolated_points.items():
    # print(curve)
    plt.plot(curve[:,0], curve[:,1], '-', label=method_name)

    
# Plot waypoints and associated index
plt.plot(waypoints[:,0], waypoints[:,1],'.')
for i in range(num_waypoints):
    plt.text(waypoints[i,0]+0.05, waypoints[i,1], str(i))
plt.legend()
plt.show()
