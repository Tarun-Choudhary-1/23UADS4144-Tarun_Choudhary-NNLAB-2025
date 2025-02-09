# Object 2 --> WAP to implement a multi-layer perceptron (MLP) network with one hidden layer using numpy in Python. Demonstrate that it can learn the XOR Boolean function. 

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def train_mlp(X, y, hidden_neurons=2, learning_rate=0.5, epochs=10000):
    np.random.seed(42)
    input_neurons = X.shape[1]
    output_neurons = y.shape[1]
    
    # Initialize weights and biases
    W1 = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
    B1 = np.random.uniform(-1, 1, (1, hidden_neurons))
    W2 = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
    B2 = np.random.uniform(-1, 1, (1, output_neurons))
    
    for epoch in range(epochs):
        # Forward propagation
        hidden_input = np.dot(X, W1) + B1
        hidden_output = sigmoid(hidden_input)
        final_input = np.dot(hidden_output, W2) + B2
        final_output = sigmoid(final_input)
        
        # Compute error
        error = y - final_output
        
        # Backpropagation
        d_output = error * sigmoid_derivative(final_output)
        d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(hidden_output)
        
        # Update weights and biases
        W2 += np.dot(hidden_output.T, d_output) * learning_rate
        B2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        W1 += np.dot(X.T, d_hidden) * learning_rate
        B1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
        
        if epoch % 1000 == 0:
            loss = np.mean(np.abs(error))
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return W1, B1, W2, B2

def predict(X, W1, B1, W2, B2):
    hidden_input = np.dot(X, W1) + B1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + B2
    final_output = sigmoid(final_input)
    return final_output

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train MLP
W1, B1, W2, B2 = train_mlp(X, y)

# Test MLP
predictions = predict(X, W1, B1, W2, B2)
print("Predictions:")
print(np.round(predictions))


''' Output --> Epoch 0, Loss: 0.5006192396930302
               Epoch 1000, Loss: 0.22070094514378455
               Epoch 2000, Loss: 0.058194278784305366
               Epoch 3000, Loss: 0.039649079093234535
               Epoch 4000, Loss: 0.03169221179058811
               Epoch 5000, Loss: 0.02706457975006713
               Epoch 6000, Loss: 0.023963665219211276
               Epoch 7000, Loss: 0.021706687427434118
               Epoch 8000, Loss: 0.019972240984918357
               Epoch 9000, Loss: 0.018587080910131483
               Predictions:
               [[0.]
                [1.]
                [1.]
                [0.]]'''