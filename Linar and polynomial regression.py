import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate some random data for demonstration
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 3 * X + 1.5 * X**2 + np.random.randn(100, 1)

# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X, y)

# Polynomial Regression
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Plotting
plt.figure(figsize=(10, 6))

# Plot original data
plt.scatter(X, y, color='blue', label='Original Data')

# Plot Linear Regression
plt.plot(X, linear_reg.predict(X), color='red', label='Linear Regression')

# Plot Polynomial Regression
plt.plot(X, poly_reg.predict(X_poly), color='green', label='Polynomial Regression')

plt.title('Linear vs Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
