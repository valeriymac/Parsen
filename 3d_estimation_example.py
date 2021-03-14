import numpy as np
import pandas as pd
from parsen import Parsen
import matplotlib.pyplot as plt

data = pd.read_csv("height_weight2.csv")
indexes = data['Index'].to_numpy()
heights_weights = data[['Height', 'Weight']].to_numpy()

X_train, y_train = heights_weights, indexes

data = pd.read_csv("height_weight_test.csv")
X_test = data[['Height', 'Weight']].to_numpy()
X_test_answers = data['Index'].to_numpy()

par = Parsen(kernel='gaussian', h=6)
par.fit(X_train, y_train, X_test)

x = np.arange(130, 210, 2)
y = np.arange(40, 180, 2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
X_t, Y_t = np.expand_dims(np.ravel(X), axis=1), np.expand_dims(np.ravel(Y), axis=1)
X_inp = np.concatenate((X_t, Y_t), axis=1)
zs = np.array(par.get_estimation(X_inp))
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='winter', edgecolor='none')
ax.set_title('h = 6')
ax.set_xlabel('Рост, см')
ax.set_ylabel('Вес, кг')
ax.set_zlabel('Density')

plt.show()
