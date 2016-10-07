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

for i in range(1, 4):
	plt.subplot(3,3,i+6)
	plt.imshow(X_test[i], cmap=plt.get_cmap('gray'))
	plt.title("Class {}".format(y_test[i]))
# show the plot
plt.show()

