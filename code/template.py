from time import sleep
import robot

arlo = robot.Robot()

sleep(1)

LEFT_FORWARD = 60
RIGHT_FORWARD = 64

TURN_LEFT_POWER = 50
TURN_RIGHT_POWER = 50

FORWARD_TIME = 2.55
TURN_TIME = 0.79

STOP_TIME = 0.5


def forward():
    arlo.go_diff(
        LEFT_FORWARD,
        RIGHT_FORWARD,
        1,
        1
    )

    sleep(FORWARD_TIME)

    arlo.stop()
    sleep(STOP_TIME)


def turn_right():
    arlo.go_diff(
        TURN_LEFT_POWER,
        TURN_RIGHT_POWER,
        1,
        0
    )

    sleep(TURN_TIME)

    arlo.stop()
    sleep(STOP_TIME)


for square in range(5):

    for side in range(4):
        forward()
        turn_right()


arlo.stop()

print("Finished")