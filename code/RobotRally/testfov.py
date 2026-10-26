import time
from camera import Camera

camera = Camera(0)

try:
    while True:
        result = camera.detect_aruco_objects(camera.get_next_frame())
        print(result)
        time.sleep(0.3)

except KeyboardInterrupt:
    pass

finally:
    camera.get_capture().close()
