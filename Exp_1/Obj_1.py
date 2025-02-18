# Object 1 --> WAP to implement the Perceptron Learning Algorithm using numpy in Python. Evaluate performance of a single perceptron for NAND and XOR truth tables as input dataset.

import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=10):
        """
        Initialize perceptron with random weights and bias.

        Parameters:
        input_size (int): Number of input features.
        lr (float): Learning rate for weight updates.
        epochs (int): Number of iterations over the dataset.
        """
        self.weights = np.random.randn(input_size)  # Random weights
        self.bias = np.random.randn()  # Random bias
        self.lr = lr
        self.epochs = epochs

    def activation(self, x):
        """Step function (Threshold at 0)."""
        return 1 if x >= 0 else 0

    def predict(self, x):
        """Compute weighted sum and apply activation function."""
        return self.activation(np.dot(x, self.weights) + self.bias)

    def train(self, X, y):
        """
        Train the perceptron using the perceptron learning rule.

        Parameters:
        X (numpy array): Input features.
        y (numpy array): Target labels.
        """
        for epoch in range(self.epochs):
            errors = 0
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction  # Compute error
                if error != 0:  # Update weights if there is an error
                    self.weights += self.lr * error * X[i]
                    self.bias += self.lr * error
                    errors += 1  # Count misclassifications
            print(f"Epoch {epoch+1}/{self.epochs}, Errors: {errors}")

# Define NAND dataset
X_NAND = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y_NAND = np.array([1, 1, 1, 0])  # NAND Outputs

# Define XOR dataset
X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y_XOR = np.array([0, 1, 1, 0])  # XOR Outputs

# Train perceptron for NAND
print("\nTraining Perceptron for NAND Gate:")
perceptron_nand = Perceptron(input_size=2, lr=0.1, epochs=10)
perceptron_nand.train(X_NAND, y_NAND)

# Test perceptron on NAND
print("\nTesting Perceptron for NAND Gate:")
for x in X_NAND:
    print(f"Input: {x}, Predicted Output: {perceptron_nand.predict(x)}")

# Train perceptron for XOR
print("\nTraining Perceptron for XOR Gate:")
perceptron_xor = Perceptron(input_size=2, lr=0.1, epochs=10)
perceptron_xor.train(X_XOR, y_XOR)

# Test perceptron on XOR
print("\nTesting Perceptron for XOR Gate:")
for x in X_XOR:
    print(f"Input: {x}, Predicted Output: {perceptron_xor.predict(x)}")

# Observations:
# The perceptron should learn NAND successfully since it is linearly separable.
# However, the perceptron will fail on XOR because XOR is not linearly separable.

#Output :
'''Training Perceptron for NAND Gate:
Epoch 1/10, Errors: 2
Epoch 2/10, Errors: 2
Epoch 3/10, Errors: 2
Epoch 4/10, Errors: 0
Epoch 5/10, Errors: 0
Epoch 6/10, Errors: 0
Epoch 7/10, Errors: 0
Epoch 8/10, Errors: 0
Epoch 9/10, Errors: 0
Epoch 10/10, Errors: 0

Testing Perceptron for NAND Gate:
Input: [0 0], Predicted Output: 1
Input: [0 1], Predicted Output: 1
Input: [1 0], Predicted Output: 1
Input: [1 1], Predicted Output: 0

Training Perceptron for XOR Gate:
Epoch 1/10, Errors: 2
Epoch 2/10, Errors: 1
Epoch 3/10, Errors: 1
Epoch 4/10, Errors: 1
Epoch 5/10, Errors: 1
Epoch 6/10, Errors: 2
Epoch 7/10, Errors: 2
Epoch 8/10, Errors: 2
Epoch 9/10, Errors: 2
Epoch 10/10, Errors: 2

Testing Perceptron for XOR Gate:
Input: [0 0], Predicted Output: 0
Input: [0 1], Predicted Output: 0
Input: [1 0], Predicted Output: 1
Input: [1 1], Predicted Output: 1'''