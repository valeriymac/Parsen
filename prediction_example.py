import pandas as pd
from parsen import Parsen


data = pd.read_csv("height_weight2.csv")
indexes = data['Index'].to_numpy()
heights_weights = data[['Height', 'Weight']].to_numpy()

X_train, y_train = heights_weights, indexes

data = pd.read_csv("height_weight_test.csv")
X_test = data[['Height', 'Weight']].to_numpy()
X_test_answers = data['Index'].to_numpy()

classifier = Parsen(kernel='gaussian', h=2.5)
classifier.fit(X_train, y_train, X_test)

print('Real answers:', X_test_answers)
print('Prediction:', classifier.predict())
