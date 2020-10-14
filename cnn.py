from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#pipelining
model=Sequential()
#cnn layer
model.add(Conv2D(32,(3,3),input_shape=(90,90,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

xy=model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 3, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'C:\Users\TEJASVI OHLAN\Desktop\family',
                                                 target_size = (90,90),
                                                 batch_size = 32)
test_set = test_datagen.flow_from_directory(r'C:\Users\TEJASVI OHLAN\Desktop\tes',
                                            target_size = (90, 90),
                                            batch_size = 32)

model.fit_generator(training_set,
                         steps_per_epoch = 2000,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 1000)

model.save('model.h5')