# Test the arlo libray by driving three distances
# at three different speeds. Everything is in cm!

from arlo import Arlo

arlo = Arlo()


arlo.drive(1, speed_cms=79, distance_cm=150).wait()
arlo.drive(1, speed_cms=17, distance_cm=70).wait()
arlo.drive(1, speed_cms=35, distance_cm=50).wait()
