# Arlo Robot Controller

from time import sleep
from template import *
from robot import Robot


class Arlo(object):
    """High(er) level API for the Arlo Robit
    
       Adds error checking"""

    
    ACCELERATION = 100   # PWM units per second^2 — tune this
    CONTROL_DT = 0.05    # seconds between speed updates

    def __init__(self, port = None):
        self.robot = Robot(port)


    def drive(self, direction : int, speed : int, distance):
        """Drive function

        Parameters:
        direction(int): Direction (1 or 0)
        speed(float): Desirec speed (40 - 127)
        distance(int): Distrance to travel in mm
        """

        # ttt: time to travel
        ttt = distance / speed

        # cs: current speed
        cs = 0.0

        target_pwm = max(self.PWM_MIN, min(speed, self.PWM_MAX))

        

        self.robot.go_diff(LEFT_FORWARD, RIGHT_FORWARD,
                           direction, direction)