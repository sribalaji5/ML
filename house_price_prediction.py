import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings

data = pd.read_csv('houseprice.csv')

X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]
y = data['price']

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    encoder = LabelEncoder()
    X['mainroad'] = encoder.fit_transform(X['mainroad'])
    X['guestroom'] = encoder.fit_transform(X['guestroom'])
    X['basement'] = encoder.fit_transform(X['basement'])
    X['hotwaterheating'] = encoder.fit_transform(X['hotwaterheating'])
    X['airconditioning'] = encoder.fit_transform(X['airconditioning'])
    X['prefarea'] = encoder.fit_transform(X['prefarea'])
    X = pd.get_dummies(X, columns=['furnishingstatus'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R^2 Score:', r2_score(y_test, y_pred))
