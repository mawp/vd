import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3 + 0.1*np.random.randn(100).astype(np.float32)

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
#optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.5, rho=0.5, epsilon=1e-08, use_locking=False, name='Adadelta')
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# Plot
print('Plotting...')
plt.plot(x_data, y_data, '.')
plt.plot(np.array([0,1]), np.array([0,1]) * sess.run(W) + sess.run(b))
plt.plot(np.array([0,1]), np.array([0,1]) * 0.1 + 0.3, 'r')
plt.show()
