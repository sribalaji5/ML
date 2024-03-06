import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        # gradient descent
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            
            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

# Example usage:
if __name__ == "__main__":
    # Sample dataset
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 1])
    
    # Create and train logistic regression model
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X, y)
    
    # Predict
    X_test = np.array([[10, 11, 12], [13, 14, 15]])
    predictions = model.predict(X_test)
    print("Predictions:", predictions)
