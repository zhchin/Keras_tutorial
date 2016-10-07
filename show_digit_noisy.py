from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 9 images as gray scale
for i in range(1, 7):
	plt.subplot(3,3,i)
	plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
	plt.title("Class {}".format(y_train[i]))

# Add noise to testing dataset
noise_factor = 0.3
X_test = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
X_test = np.clip(X_test, 0., 1.)

for i in range(1, 4):
	plt.subplot(3,3,i+6)
	plt.imshow(X_test[i], cmap=plt.get_cmap('gray'))
	plt.title("Class {}".format(y_test[i]))
# show the plot
plt.show()

