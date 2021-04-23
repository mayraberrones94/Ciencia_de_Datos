
#-----------------------------------------------------------------------------------------------------------------------------------------------
#Adam con 160x400 - 32 64 64 - 240 - Dropout 0.2
#-----------------------------------------------------------------------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (80, 200, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units= 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/home/mayra/Documentos/INCAN/Train-80',
                                                 target_size = (80, 200),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('/home/mayra/Documentos/INCAN/Dev-80',
                                            target_size = (80, 200),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000)

classifier.save('Modelo_1211-sdrop.h5')
print("Fin de Modelo_1211-3")

#-----------------------------------------------------------------------------------------------------------------------------------------------
#Adam con 1600x400 - 32 32 32 - 380 - Dropout 0.2
#-----------------------------------------------------------------------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (80, 200, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units= 380, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/home/mayra/Documentos/INCAN/Train-80',
                                                 target_size = (80, 200),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('/home/mayra/Documentos/INCAN/Dev-80',
                                            target_size = (80, 200),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000)

classifier.save('Modelo_1231sdrop.h5')
print("Fin de Modelo_1231-3")



#-----------------------------------------------------------------------------------------------------------------------------------------------
#SGD con 160x400 - 32 32 32 - 128 - Dropout 0.2
#-----------------------------------------------------------------------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (80, 200, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units= 128, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'SGD', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/home/mayra/Documentos/INCAN/Train-80',
                                                 target_size = (80, 200),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('/home/mayra/Documentos/INCAN/Dev-80',
                                            target_size = (80, 200),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000)

classifier.save('Modelo_1215-sdrop.h5')
print("Fin de Modelo_1215-3")

#-----------------------------------------------------------------------------------------------------------------------------------------------
#SGD con 160x400 - 32 32 32 - 380 - Dropout 0.2
#-----------------------------------------------------------------------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (80, 200, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units= 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'SGD', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/home/mayra/Documentos/INCAN/Train-80',
                                                 target_size = (80, 200),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('/home/mayra/Documentos/INCAN/Dev-80',
                                            target_size = (80, 200),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000)

classifier.save('Modelo_1315-sdrop.h5')
print("Fin de Modelo_1315-3")


#-----------------------------------------------------------------------------------------------------------------------------------------------
#Adadelta con 160x400 - 32 64 64 - 240 - Dropout 0.2
#-----------------------------------------------------------------------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (80, 200, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units= 240, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'Adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/home/mayra/Documentos/INCAN/Train-80',
                                                 target_size = (80, 200),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('/home/mayra/Documentos/INCAN/Dev-80',
                                            target_size = (80, 200),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000)

classifier.save('Modelo_1226-sdrop.h5')
print("Fin de Modelo_1226-3")


#-----------------------------------------------------------------------------------------------------------------------------------------------
#Adadelta con 160x400 - 32 64 64 - 380 - Dropout 0.2
#-----------------------------------------------------------------------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Activation
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (80, 200, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units= 380, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'Adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/home/mayra/Documentos/INCAN/Train-80',
                                                 target_size = (80, 200),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('/home/mayra/Documentos/INCAN/Dev-80',
                                            target_size = (80, 200),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000)

classifier.save('Modelo_1136-sdrop.h5')
print("Fin de Modelo_1136-3")