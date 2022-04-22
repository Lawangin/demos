import keras.models
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img

model = keras.models.load_model('my_model.h5')
img_width, img_height = 28, 28
img = load_img('num3.jpg', True, target_size=(img_width, img_height))
img_arr = img_to_array(img)
img_arr = img_arr.astype('float32') / 255
print('Before expanding dims:', img_arr.shape)
img_arr = np.expand_dims(img_arr, axis=0)
print('After expanding dims:', img_arr.shape)

pred_range = model.predict(img_arr).argmax(axis=1)
print('predict:', pred_range)

plt.imshow(img)
plt.show()