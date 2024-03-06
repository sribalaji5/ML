import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('carprice.csv')
X = data[['carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'horsepower', 'citympg', 'highwaympg']]
y = data['price']
encoder = LabelEncoder()
data['CarName'] = encoder.fit_transform(data['CarName'])
X = pd.concat([X, pd.get_dummies(data[['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']])], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R^2 Score:', r2_score(y_test, y_pred))