import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model, datasets


n_per_line = 100
n_lines = 3

X = np.ndarray((n_per_line * n_lines, 1))
y = np.ndarray(n_per_line * n_lines)

for iline in range(n_lines):
    slope = 100. * np.random.random()
    for ii in range(n_per_line):
        randx = np.random.random() * 10
        X[ii+iline*n_per_line][0] = randx
        y[ii+iline*n_per_line] = slope*randx + 50. * np.random.random()

# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
line_X = np.ndarray((1000,1))
for ii in range(1000):
    line_X[ii][0] = 10. * np.random.random()

line_y_ransac = ransac.predict(line_X)

lw = 2
plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
            label='Outliers')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
         label='RANSAC regressor')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.savefig("./RANSAC_test.png")
