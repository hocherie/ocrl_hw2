#!/bin/bash
roslaunch ocrl ocrl.launch&
sleep 7
rosrun ocrl trajectory_optimizer.py&
sleep 1
rosrun ocrl lqr2.py
