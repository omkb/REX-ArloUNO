from time import sleep
import robot

arlo = robot.Robot()

sleep(1)

LEFT_FORWARD = 54
RIGHT_FORWARD = 54

TURN_LEFT_POWER = 50
TURN_RIGHT_POWER = 50

FORWARD_TIME = 2.91
TURN_TIME = 0.938

OUTER = 55
INNER = 0

RIGHT_LOOP_TIME = 7.2
LEFT_LOOP_TIME = 7.2

STOP_TIME = 0.5


def forward():
    arlo.go_diff(LEFT_FORWARD, RIGHT_FORWARD, 1, 1)

    sleep(FORWARD_TIME)

    arlo.stop()
    sleep(STOP_TIME)


def turn_right():
    arlo.go_diff(TURN_LEFT_POWER, TURN_RIGHT_POWER, 1, 0)

    sleep(TURN_TIME)

    arlo.stop()
    sleep(STOP_TIME)

def square():
    for square in range(5):
        for side in range(4):
            forward()
            turn_right()


def figure_eight():
    while True:
        # Right circle
        arlo.go_diff(OUTER, INNER, 1, 1)
        sleep(RIGHT_LOOP_TIME)

        # WITHOUT STOPPING: immediately change curvature
        arlo.go_diff(INNER, OUTER, 1, 1)
        sleep(LEFT_LOOP_TIME)


figure_eight()

arlo.stop()

print("Finished")
