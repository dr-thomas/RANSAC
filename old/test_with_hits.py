import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model, datasets

import csv


filepath = "../csv/train_0011.csv"

X_data = []
y_data = []
with open(filepath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        used_row = True
        for ii, xx in enumerate(row):
            if ii == 0:
                if int(xx) < 2:
                    used_row = False
                    break
            if ((ii-1)%4) == 1:
                X_data.append(float(xx))
            elif ((ii-1)%4) == 2:
                y_data.append(float(xx))
        if used_row:
            n_nonzero = 0
            for ii in range(len(X_data)):
                if X_data[ii] == 0 and y_data[ii] == 0:
                    continue
                n_nonzero += 1
            if n_nonzero > 50 and n_nonzero<400:
                line_count += 1
            if line_count == 2:
                break
            else:
                X_data.clear()
                y_data.clear()

X = np.ndarray((n_nonzero,1))
y = np.ndarray(n_nonzero)
idata = 0
for ii in range(len(X_data)):
    if X_data[ii] == 0 and y_data[ii] == 0:
        continue 
    X[idata][0] = X_data[ii]
    y[idata] = y_data[ii] 
    idata += 1

# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated model
line_X = np.ndarray((1000,1))
xline_point = X.min()
iline = 0
while xline_point < X.max():
    if iline < 1000:
        line_X[iline][0] = xline_point
        iline += 1
    xline_point += (X.max() - X.min())/1000

line_y_ransac = ransac.predict(line_X)

lw = 2
plt.scatter(X[inlier_mask], y[inlier_mask], color='cornflowerblue', marker='.',
            label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color='yellowgreen', marker='.',
            label='Outliers')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
         label='RANSAC regressor')
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("./RANSAC_test1.png")

X = X[outlier_mask]
y = y[outlier_mask]

ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)




# Predict data of estimated model
line_X = np.ndarray((1000,1))
xline_point = X.min()
iline = 0
while xline_point < X.max():
    if iline < 1000:
        line_X[iline][0] = xline_point
        iline += 1
    xline_point += (X.max() - X.min())/1000

line_y_ransac = ransac.predict(line_X)

plt.figure()
lw = 2
plt.scatter(X[inlier_mask], y[inlier_mask], color='cornflowerblue', marker='.',
            label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color='yellowgreen', marker='.',
            label='Outliers')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
         label='RANSAC regressor')
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("./RANSAC_test2.png")
