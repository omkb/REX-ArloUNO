from time import sleep
import robot

arlo = robot.Robot()
sleep(1)

FORWARD = 54
TURN = 48

SAFE_DISTANCE = 500
DANGER_DISTANCE = 250


def read_sensors():
    return (
        arlo.read_front_ping_sensor(),
        arlo.read_left_ping_sensor(),
        arlo.read_right_ping_sensor()
    )


def choose_action(front, left, right):
    # Very close obstacle straight ahead
    if front < DANGER_DISTANCE:
        if left > right:
            return "sharp_left"
        else:
            return "sharp_right"

    # Very close obstacle on one side
    elif left < DANGER_DISTANCE:
        return "sharp_right"

    elif right < DANGER_DISTANCE:
        return "sharp_left"

    # Obstacle ahead, but not critically close
    elif front < SAFE_DISTANCE:
        if left > right:
            return "left"
        else:
            return "right"

    # Obstacle somewhat close on one side
    elif left < SAFE_DISTANCE:
        return "right"

    elif right < SAFE_DISTANCE:
        return "left"

    # Nothing nearby
    else:
        return "forward"


def execute_action(action):
    if action == "forward":
        arlo.go_diff(FORWARD, FORWARD, 1, 1)

    elif action == "left":
        # Curve left
        arlo.go_diff(TURN, FORWARD, 1, 1)

    elif action == "right":
        # Curve right
        arlo.go_diff(FORWARD, TURN, 1, 1)

    elif action == "sharp_left":
        # Rotate left on the spot
        arlo.go_diff(TURN, TURN, 0, 1)

    elif action == "sharp_right":
        # Rotate right on the spot
        arlo.go_diff(TURN, TURN, 1, 0)

    else:
        arlo.stop()


print("Running obstacle avoidance...")


while True:
    front, left, right = read_sensors()

    action = choose_action(front, left, right)

    print(
	"Front:", front,
	"Left:", left,
	"Right:", right,
	"Action:", action
    )

    execute_action(action)

    # Small pause so we do not spam the Arduino
    sleep(0.1)

