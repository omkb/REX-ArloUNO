import numpy as np
import matplotlib.pyplot as plt

points_old = np.array([
    [0, -22.5],
    [-5.18648092, 7.04827263],
    [-6.63222975, 37.01341589],
    [-6.08418281, 67.00840955],
    [-17.75990502, 94.64312523],
    [-44.39512646, 108.4476527],
    [-54.05905442, 80.0467989],
    [-81.63285925, 91.86565194],
    [-105.99236336, 109.37606483],
    [-112.6430797, 138.62957697],
    [-116.96874955, 168.31608199],
    [-122.66521502, 197.7702885],
    [-119.89033969, 227.64168061],
    [-125.14273452, 257.17830785],
    [-105.37205377, 279.74199954],
    [-112.49548534, 308.88400913],
    [-89.18302651, 327.76599163],
    [-65.87056768, 346.64797414],
    [-54.69943631, 374.49049241],
    [-30.37635501, 392.0514638],
    [-3.86833809, 406.09870691],
    [0, 400],
])

obstacles_old = [
    (6, -56.94008560181654, 218.99234929672883, 63.07886552931954),
    (7, -5.985835258154749, 232.16319622730995, 63.07886552931954),
    (10, 49.6818105651655, 213.38233086900408, 63.07886552931954),
]

points = np.array([
    [0, -22.5],
    [54.86074582, 228.31957374],
    [25.40139823, 233.98939203],
    [-97.38289628, 334.13263708],
    [0, 400],
])

obstacles = np.array([
    (7, -113.37504145613694, 216.71924063474788, 63.07886552931954),
    (6, -38.13009335021692, 200.37515597061144, 63.07886552931954),
    (10, 53.51471050562822, 293.7571613344797, 63.07886552931954),
])

fig, ax = plt.subplots()

# plot the path/points
ax.plot(points[:, 0], points[:, 1],
        marker='o', markerfacecolor='red', markeredgecolor='red',
        linestyle='-', color='gray', label='point')

# plot obstacles as yellow circles with id labels
for obs_id, x, y, radius in obstacles:
    circle = plt.Circle((x, y), radius,
                         facecolor='yellow', edgecolor='black',
                         alpha=0.6, label='obstacle' if obs_id == obstacles[0][0] else None)
    ax.add_patch(circle)
    ax.annotate(str(obs_id), (x, y), ha='center', va='center', fontweight='bold')

ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.grid()
ax.legend()
ax.set_aspect('equal')
ax.autoscale_view()
plt.show()