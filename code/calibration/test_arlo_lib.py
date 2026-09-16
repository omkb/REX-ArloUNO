# Test the arlo libray by driving three distances
# at three different speeds. Everything is in cm!

from arlo import Arlo

arlo = Arlo()

#arlo.drive(1, speed_cms=60, distance_cm=300).wait()
#arlo.turn(-90, 90).wait()
#arlo.drive(1, speed_cms=40, distance_cm=150).wait()
#arlo.turn(90, 90).wait()

arlo.turn(-90, 90).wait()
arlo.drive(0, speed_cms=40, distance_cm=150).wait()
arlo.turn(90, 90).wait()
arlo.drive(0, speed_cms=40, distance_cm=300).wait()
