import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def read_driving_log(driving_log):
    lines = []
    with open(driving_log) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

"""
img_dirs = ['../IMG/',
            'goodLoop/IMG/',
            'goodLoop+badLoop/IMG/',
            'backwardsLoop/IMG/',
            'Recovery/IMG/',
            'bridge/IMG/',
            'lvl2/firstRound/IMG/',
            'lvl2/backwards/IMG/', 
            'lvl2/secondRound/IMG/',
            ]
driving_logs = ['./driving_log.csv',
                'goodLoop/driving_log.csv',
                'goodLoop+badLoop/driving_log.csv',
                'backwardsLoop/driving_log.csv',
                'Recovery/driving_log.csv',
                'bridge/driving_log.csv',
                'lvl2/firstRound/driving_log.csv',
                'lvl2/backwards/driving_log.csv',
                'lvl2/secondRound/driving_log.csv',
                ]

images = []
measurements = []
for img_dir, driving_log in zip(img_dirs, driving_logs):
    lines = read_driving_log(driving_log) 
    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = img_dir + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
print("Num samples: {}".format(len(images)))
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
"""
samples = []
with open('./IMG/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import sklearn
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                #center_image = batch_sample[0]
                #center_angle = float(batch_sample[1])
                images.append(center_image)
                angles.append(center_angle)
                # add augmentation
                images.append(cv2.flip(center_image, 1))
                angles.append(center_angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#ch, row, col = 3, 80, 320  # Trimmed image format
model = Sequential()
#model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(ch, row, col),
#                    output_shape=(ch, row, col)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4, verbose=1)
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
					validation_data=validation_generator, \
            			nb_val_samples=len(validation_samples), nb_epoch=4, verbose=1)


model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
