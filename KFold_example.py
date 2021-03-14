import pandas as pd
import matplotlib.pyplot as plt
from parsen import Parsen
from parameter_selection import KFold_search

data = pd.read_csv("height_weight2.csv")
qualities = data['Index'].to_numpy()
components = data[['Height', 'Weight']].to_numpy()

X_train, y_train = components, qualities

data = pd.read_csv("height_weight_test.csv")
X_test = data[['Height', 'Weight']].to_numpy()

classifier = Parsen(kernel='gaussian', h=2.5)
classifier.fit(X_train, y_train, X_test)

print(classifier.predict())

loo = KFold_search(5, neighbours=False, kernel='gaussian')
sums = loo.get_parameter(X_train, y_train, start=0.5, stop=5.5, step=1)
plt.plot(range(0.5, 5.5, 1), sums)
plt.show()
