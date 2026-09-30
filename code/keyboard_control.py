"""Keyboard control for the Arlo robot.

Controls (latching: the robot keeps doing the last command until you change it):
    W / Up     forward
    S / Down   backward
    A / Left   turn left (on the spot)
    D / Right  turn right (on the spot)
    Space / X  stop
    + / -      increase / decrease motor power
    P          read front ping sensor
    Q          quit (stops the robot first)

Run:  python keyboard_control.py [--port /dev/ttyACM0]
"""

import argparse
import curses
import sys
from pathlib import Path

# Make the robot module importable. Adjust if your Arlo folder lives elsewhere.
here = Path(__file__).resolve().parent
for candidate in (here / "../Arlo/python", here / "../../Arlo/python", here / "../../../Arlo/python"):
    if candidate.exists():
        sys.path.append(str(candidate.resolve()))
        break

import robot  # noqa: E402

MIN_POWER = 40   # below 40 the Arlo may blow a fuse
MAX_POWER = 127
STEP = 10

# state name -> (dirLeft, dirRight); 1 = forward, 0 = reverse
MOTIONS = {
    "forward": (1, 1),
    "backward": (0, 0),
    "left": (0, 1),
    "right": (1, 0),
}

KEYMAP = {
    ord("w"): "forward", curses.KEY_UP: "forward",
    ord("s"): "backward", curses.KEY_DOWN: "backward",
    ord("a"): "left", curses.KEY_LEFT: "left",
    ord("d"): "right", curses.KEY_RIGHT: "right",
    ord(" "): "stop", ord("x"): "stop",
}


def apply(arlo, state, power):
    """Send the command for the given state to the robot."""
    if state == "stop":
        arlo.stop()
    else:
        dir_left, dir_right = MOTIONS[state]
        arlo.go_diff(power, power, dir_left, dir_right)


def draw(stdscr, state, power, message):
    stdscr.erase()
    lines = [
        "Arlo keyboard control",
        "",
        "  W/Up: forward    S/Down: backward",
        "  A/Left: turn left    D/Right: turn right",
        "  Space/X: stop    +/-: power    P: front ping    Q: quit",
        "",
        f"State: {state}",
        f"Power: {power}",
        f"Info:  {message}",
    ]
    for i, line in enumerate(lines):
        try:
            stdscr.addstr(i, 0, line)
        except curses.error:
            pass  # terminal too small
    stdscr.refresh()


def main(stdscr, arlo):
    curses.curs_set(0)
    stdscr.keypad(True)
    stdscr.timeout(100)  # getch waits at most 100 ms

    state = "stop"
    power = 40
    message = ""
    draw(stdscr, state, power, message)

    try:
        while True:
            key = stdscr.getch()
            if key == -1:
                continue  # no key pressed

            if key in (ord("q"), ord("Q")):
                break

            lower = ord(chr(key).lower()) if 0 <= key < 256 else key

            if lower in KEYMAP:
                new_state = KEYMAP[lower]
                if new_state != state:  # only send when something changes
                    state = new_state
                    apply(arlo, state, power)
                message = ""
            elif lower in (ord("+"), ord("=")):
                power = min(MAX_POWER, power + STEP)
                if state != "stop":
                    apply(arlo, state, power)
            elif lower in (ord("-"), ord("_")):
                power = max(MIN_POWER, power - STEP)
                if state != "stop":
                    apply(arlo, state, power)
            elif lower == ord("p"):
                message = f"Front distance: {arlo.read_front_ping_sensor()} mm"

            draw(stdscr, state, power, message)
    finally:
        arlo.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keyboard control for the Arlo robot")
    parser.add_argument("--port", default="/dev/ttyACM0", help="serial port of the Arduino")
    args = parser.parse_args()

    arlo = robot.Robot(port=args.port)
    try:
        curses.wrapper(main, arlo)
    except KeyboardInterrupt:
        pass
    finally:
        arlo.stop()
        print("Stopped.")
