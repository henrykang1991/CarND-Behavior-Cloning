import csv
import cv2
import numpy as np
import sklearn
import random

lines=[]
with open('Data/driving_log.csv')as csvfile:
	reader=csv.reader(csvfile)
	for line in reader:
		lines.append(line)
from sklearn.model_selection import train_test_split
train_samples, validation_samples=train_test_split(lines,test_size=0.2)
images=[]
measurements=[]
def generator (samples, batch_size=32):
	num_samples=len(samples)
	while 1:
		random.shuffle(samples)
		for offset in range(0,num_samples,batch_size):
			batch_samples=samples[offset:offset+batch_size]
			img_center=[]
			img_left=[]
			img_right=[]
			images=[]
			measurements=[]
			for batch_sample in batch_samples :
				steering_center = float(batch_sample[3])
				# create adjusted steering measurements for the side camera imagess
				correction = 0.35 # this is a parameter to tune
				steering_left = steering_center + correction
				steering_right = steering_center - correction
				# read in images from center, left and right cameras
				path = 'Data/IMG/' # fill in the path to your training IMG directory
				img_center=cv2.imread(path+batch_sample[0].split('\\')[-1])
				img_left=cv2.imread(path+batch_sample[1].split('\\')[-1])
				img_right=cv2.imread(path+batch_sample[2].split('\\')[-1])

				# add images and angles to data set
			images.extend([img_center,img_left,img_right])
			measurements.extend([steering_center,steering_left,steering_right])
			augmented_images, augmented_measurements=[],[]
			for image, measurement in zip(images, measurements):
				augmented_images.append(image)
				augmented_measurements.append(measurement)
				#image_flipped=np.fliplr(image)
				image_flipped=cv2.flip(image,1)
				measurement_flipped=-measurement
				augmented_images.append(image_flipped)
				augmented_measurements.append(measurement_flipped)

			X_train=np.array(augmented_images)
			y_train=np.array(augmented_measurements)
			yield sklearn.utils.shuffle(X_train,y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from keras.layers import Cropping2D
model=Sequential()
model.add(Lambda(lambda x:x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
model.compile(loss='mse', optimizer='adam')
#model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
from keras.models import Model
model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data =
    validation_generator,
    nb_val_samples = len(validation_samples),
    nb_epoch=5, verbose=1)
model.save('model.h5')

