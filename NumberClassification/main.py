import logging
import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img

# suppress warnings and only display errors
tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 7
BATCH_SIZE = 1

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

"""
Load data
"""
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

plt.imshow(train_images[1].reshape(28, 28, 1))
plt.show()
"""
preprocess (standardize) data
"""
# mean = np.mean(train_images)
# stddev = np.std(train_images)
# train_images = (train_images - mean) / stddev
# test_images = (test_images - mean) / stddev

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

print(train_images.shape)

# one-hot encode labels
# zero,	  one,	    two
# 1,		0,		0
# 0,		1,		0
# 0,		0,		1
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# object used to initialize weights
initializer = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)

"""
create model
"""
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
# model.add(keras.layers.Dense(25, activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(keras.layers.Dense(25, activation='relu'))
# model.add(keras.layers.Dense(10, activation='sigmoid', kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(keras.layers.Dense(10, activation='softmax'))

"""
train model
"""
# stochastic gradient descent (SGD) with learning rate of 0.01
# opt = keras.optimizers.SGD(learning_rate=0.01)
opt = keras.optimizers.Adam(lr=0.01)

# model.compile(loss='mean_squared_error', optimizer=opt, metrics=['acc'])
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2,
                    verbose=1, shuffle=True)

model.save('my_model.h5')

"""
plot results
"""
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

"""
test model
"""
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)

img_width, img_height = 28, 28
img = load_img('num5.jpg', False, target_size=(img_width, img_height))
img_arr = img_to_array(img)
print('Before expanding dims:', img_arr.shape)
img_arr = np.expand_dims(img_arr, axis=0)
print('After expanding dims:', img_arr.shape)

pred_range = model.predict(img_arr).argmax(axis=1)
print('predict:', img_arr)
