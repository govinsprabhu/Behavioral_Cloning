import os
import csv
from keras.models import Sequential, Model
from keras.layers import Lambda, Dense, Conv2D, Activation, Flatten, Cropping2D, MaxPool2D, Dropout
import matplotlib.pyplot as plt

samples = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # skipping the headers
    next(reader, None)
    for line in reader:
        samples.append(line)

print(len(samples))

from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '/opt/carnd_p3/data/IMG/' + batch_sample[0].split('/')[-1]
                # print(name)
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # print(center_image.shape)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            # print(X_train.shape, y_train.shape, offset)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

# Preprocess incoming data, centered around zero with small standard deviation

model = Sequential()
model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 127.5 - 1.,
                 input_shape=(row, col, ch),
                 output_shape=(row, col, ch)))

# incoming 80x320x3
model.add(Conv2D(24, (5, 5), strides=(1, 1), padding='same'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Activation('relu'))


# incoming 40x180x24
model.add(Conv2D(36, (5, 5), strides=(1, 1), padding='same'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Activation('relu'))


# incoming 20x90x36
model.add(Conv2D(48, (5, 5), strides=(1, 1), padding='same'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# incoming 10x45x48
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# incoming 5x22x64
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
model.add(MaxPool2D((2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# incoming 3x11x64
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.summary()
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
                                     validation_data=validation_generator, validation_steps=len(validation_samples),
                                     epochs=1, verbose=1)

print(history_object.history.keys())
model.save('my_model.h5')

# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('Model Mean square error loss')
# plt.ylabel('Mean square error loss')
# plt.xlabel('Epoch')
# plt.legend(['Training Set', 'Validation loss'], loc = 'upper right')
# plt.show()





