import numpy as np

#image pixels 1640*1232

X = 14.4  # cm

distances = np.array([
    60, 70, 80, 90, 100, 110, 120, 130,
    140, 150, 160, 170, 180, 190, 200, 210
])

pixel_heights = np.array([
    317, 272, 236, 205, 188, 169, 155, 141,
    133, 123, 116, 107, 100, 97, 92, 88
])

f_values = pixel_heights * distances / X

print("Focal lengths:")
print(f_values)

print("Mean:", np.mean(f_values))
print("Standard deviation:", np.std(f_values, ddof=1))
