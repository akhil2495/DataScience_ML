import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

tr_x = np.array([0,2,3,5])
tr_y = np.array([1,4,9,16])
te_x = np.array([1,4])
te_y = np.array([3,12])
b = []
for i in range(4):
    a = np.polyfit(tr_x, tr_y, i)
    b.append(np.poly1d(a))
c = [[0,0,0], [0.5,0.5,0.5], [0.75,0.75,0.75], [0.9,0.9,0.9]]

plt.plot(tr_x, tr_y, 'r*', label='train data', markersize=12)
plt.plot(te_x, te_y, 'bo', label='test data', markersize=12)
d = np.sort(np.concatenate((tr_x, te_x)))
e = np.array([float(i)/10 for i in range(60)])
print e
for i in range(4):
    plt.plot(e, b[i](e), color =c[3-i], label='degree' + str(i))
plt.title('Regression Curves')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.tick_params(labelsize=12)
plt.legend()
plt.show()

# part-2
bias = []
var = []
total_error = []
train_error = []
test_error = []
for deg in [0,1,2,3]:
    bias.append((np.mean(np.array([b[deg][i] for i in d])) - np.mean(np.concatenate((tr_y, te_y))))**2)
    var.append(np.mean(np.array([(b[deg][i]-np.mean(np.array([b[deg][i] for i in d])))**2 for i in d])))
    total_error.append(np.sum(np.abs(np.array(np.concatenate((tr_y, te_y)))-np.array([b[deg][i] for i in d]))))
    train_error.append(np.sum(np.abs(tr_y - np.array([b[deg][i] for i in tr_x.tolist()]))))
    test_error.append(np.sum(np.abs(te_y - np.array([b[deg][i] for i in te_x.tolist()]))))
plt.plot(bias, color=[0,0,0], label='bias')
plt.plot(var, color=[0.7,0.7,0.7], label='var')
plt.plot(total_error, 'g', label='total error')
plt.plot(train_error, 'r', label='train error')
plt.plot(test_error, 'b', label='test error')
plt.xlabel('Degree of polynomial regressor')
plt.ylabel('Legend')
plt.legend(loc='upper right')
plt.show()
