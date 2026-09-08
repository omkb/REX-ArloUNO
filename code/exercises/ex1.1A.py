from time import sleep

import robot

# Create a robot object and initialize
arlo = robot.Robot()

print("Running ...")


def drive():
    """Drives forward, stops, turns 90 degress clockwise, stop"""
    # send a go_diff command to drive forward
    leftSpeed = 60
    rightSpeed = 60
    minute = 2.55
    ninety = 0.79
    
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))

    # Wait a bit while robot moves forward
    sleep(minute)

    # send a stop command
    print(arlo.stop())

    # Wait a bit before next command
    sleep(0.041)

    # send a go_diff command to turn 
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))

    #Wait a bit while robot moves backwards
    sleep(ninety)

    # send a stop command
    print(arlo.stop())

    # Wait a bit before next command
    sleep(0.041)


for i in range(4):
	drive()

print("Finished")

