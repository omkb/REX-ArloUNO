# Exercise 1: Drive forward, stop, turn 90 deg, stop. Repeat 4 times.

import sys, os
from time import sleep

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from arlo import Direction, Arlo

# Create an arlo object - arlo acts as wrapper for Robot class
arlo = Arlo()

print("Start")

response = arlo.robot.go_diff(80, 80, 1, 1)
#print(repr(response))
#sleep(1)
#arlo.robot.stop()
arlo.drive(Direction.BACKWARD, speed_cms=30, distance_cm=300).wait()

print("End")
