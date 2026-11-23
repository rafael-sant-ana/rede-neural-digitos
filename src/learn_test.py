import learn
import numpy as np

# --- TESTED, OK ---
# W1, b1, W2, b2 = learn.initialize_parameters(784, 4, 10)

# print("W1 = ", W1)
# print("b1 = ", b1)
# print("W2 = ", W2)
# print("b2 = ", b2)

# --- TESTED, OK ---
# a = learn.relu(np.array([[-1,0,1], [-2,0,2]]))

# print("a = ", a)

# --- TESTED, OK ---
# a = learn.softmax(np.array([[1,2,3], [4,5,6]]))
# print("a = ", a)

# --- TESTED, OK ---
# X = np.random.randn(2, 784)
# parameters = learn.initialize_parameters(784, 4, 10)
# Z1, A1, Z2, A2 = learn.forward(X, parameters)

# print(Z1.shape, A1.shape, Z2.shape, A2.shape)

# --- TESTED, OK ---
# y_true = np.array([[0,1,0], [1,0,0]])
# y_pred = np.array([[0.1, 0.6, 0.3], [0.3, 0.4, 0.3]])
# loss = learn.cross_entropy_loss(y_true, y_pred)

# print(loss)

# --- TESTED, OK ---
# X = np.random.randn(2, 784)
# y = np.array([
#     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
# ])
# parameters = learn.initialize_parameters(784, 4, 10)
# parameters, loss = learn.gradient_descent_step(X, y, parameters, learning_rate=0.01)

# print("loss = ", loss)

# --- TESTED, OK ---
# y_true = np.array([[0, 1, 0], [1, 0, 0]])
# y_pred = np.array([[0.1, 0.6, 0.3], [0.3, 0.4, 0.3]])
# acc = learn.accuracy(y_true, y_pred)

# print("acc = ", acc)
