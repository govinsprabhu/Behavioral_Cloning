import csv
import cv2
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


lines = []
def get_data(path):
    # getting driving_log, and loading the image paths
    with open(path+'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader,None)
        for line in reader:
            lines.append(line)
    return lines

def generator(data, path, batch_size = 32):
    num_samples = len(data)
    angle_adjustment = 0.1
    image_path = path +'IMG/'
    while 1:
        data = shuffle(data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset + batch_size]
            images = []
            angles = []
            for line in batch_samples:
                # center image
                # converting to RGB and adding both flipped and original
                center_image = cv2.imread(image_path + line[0].split('/')[-1])
                center_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                images.append(center_image_rgb)
                angles.append(float(line[3]))
                images.append(cv2.flip(center_image_rgb, 1))
                angles.append(-float(line[3]))

                # left image
                # converting to RGB and adding both flipped and original
                left_image = cv2.imread(image_path + line[1].split('/')[-1])
                left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                images.append(left_image_rgb)
                angles.append(float(line[3]) + angle_adjustment)
                images.append(cv2.flip(left_image_rgb, 1))
                angles.append(-(float(line[3]) + angle_adjustment))

                # right image
                # converting to RGB and adding both flipped and original
                right_image = cv2.imread(image_path + line[2].split('/')[-1])
                right_image_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                images.append(right_image_rgb)
                angles.append(float(line[3]) - angle_adjustment)
                images.append(cv2.flip(right_image_rgb, 1))
                angles.append(-(float(line[3]) - angle_adjustment))

            # converting to numpy array
            #print(len(images), len(angles))
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)



def get_model():
    # creating model based on NVIDIA paper
    model = Sequential()
    # applying normalization to the image
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

    # Cropping the image, 70 from top, 25 from bottom
    # Input 160x320x3
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    # Applying 24 filter of sizes (5,5) of strides of 2 with relu activation
    # input 65x320x3
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))

    # Applying 36 filter of sizes (5,5) of strides of 2 with relu activation
    # input 31x158x24
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))

    # Applying 48 filter of sizes (5,5) of strides of 2 with relu activation
    # input 14x77x36
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))

    # Applying 64 filter of sizes (5,5) of strides of 1 with relu activation
    # input 5x37x48
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Applying 64 filter of sizes (5,5) of strides of 2 with relu activation
    # input 3x35x64
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # input 1x33x64
    model.add(Flatten())

    # input 2112
    model.add(Dense(100))

    # input 100
    model.add(Dense(50))

    # input 50
    model.add(Dense(10))

    # input 10
    model.add(Dense(1))

    # using adam optimization, and mean square error
    model.compile('adam', 'mse')

    model.summary()
    return model

path = 'C:/Users/609600403/Documents/ML/project/CarND-Behavioral-Cloning-P3-master/data/'

# loading the image paths from csv
lines = get_data(path)
print(len(lines))
# Splitting train and validation ,used 20% of data for validation
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Getting training and validation using generator function, used batch of 32
train_generator = generator(train_samples, path, batch_size=32)
validation_generator = generator(validation_samples, path, batch_size=32)

# getting model
model = get_model()
# when you are loading the model

#model = load_model('model-4.h5')


# training the model using generator
model.fit_generator(train_generator, steps_per_epoch=4*len(train_samples),validation_data=validation_generator, validation_steps=len(validation_samples),epochs=1, verbose=1)
# Saving the model
model.save('model-5.h5')