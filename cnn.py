import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


# load local data
def listdir_fullpath(d):
    return np.sort([os.path.join(d, f) for f in os.listdir(d)])

def load_images(paths):
    data = []
    for imagePath in paths:
        # load the image, pre-process it, and store it in the data list
        image = load_img(
            imagePath, grayscale=False, color_mode='rgb', target_size=(1500,1500,3),
            interpolation='nearest'
        )
        image = img_to_array(image)
        data.append(image)
    data = np.array(data, dtype="float") / 255.0
    return data

# input and output are both images of size 1500, 1500,3
test_input = listdir_fullpath("./road_segmentation/testing/input/")
test_output = listdir_fullpath("./road_segmentation/testing/output/")
train_input = listdir_fullpath("./road_segmentation/training/input/")
train_output = listdir_fullpath("./road_segmentation/training/output/")

test_input_data = load_images(test_input)
test_output_data = load_images(test_output)
train_input_data = load_images(train_input)
train_output_data = load_images(train_output)

#plt.imshow(test_input_data[1])
#plt.imshow(test_output_data[1])

print(test_input_data.shape)
print(test_output_data.shape)

model = models.Sequential()
model.add(layers.Conv2D(1500, (3, 3), activation='relu', input_shape=(1500, 1500, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.UpSampling2D(size=(4,4)))
model.add(layers.ZeroPadding2D((8,8)))


model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_input_data, train_output_data, epochs=10,
                    validation_data=(test_input_data, test_output_data))