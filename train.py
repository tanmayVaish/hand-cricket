from model import Model_Making as mod
from capture import Capture as cp
from preprocess import Preprocess 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np


m = mod.Modelling();

# Capturing -----------------------------------------------------------------------------------------
num_samples = 200

data = cp.Collect(num_samples)
images = Preprocess.convert(data)
# ---------------------------------------------------------------------------------------------------


# Preprocessing -------------------------------------------------------------------------------------

blur_images = Preprocess.gaussian(images) # contains list of images of all classess in ascending order
seg_images = Preprocess.segmentation(blur_images)
morph_images = Preprocess.morphology(seg_images,images)


# ---------------------------------------------------------------------------------------------------


# Creating Labels------------------------------------------------------------------------------------

labels = []
labels += [1]*num_samples
labels += [2]*num_samples
labels += [3]*num_samples
labels += [4]*num_samples
labels += [5]*num_samples
labels += [0]*num_samples

# ---------------------------------------------------------------------------------------------------

trainX = []
testX = []
trainY = []
testY = []

for i in range(6):
    (t1, t2, t3, t4) = train_test_split(morph_images[(i*num_samples):((i+1)*num_samples)], labels[(i*num_samples):((i+1)*num_samples)], test_size=0.2, random_state=50)
    trainX += t1
    testX += t2
    trainY += t3
    testY += t4
    
print(trainY)
print(testY)

trainX = np.array(trainX,dtype="float") / 255.0
testX = np.array(testX,dtype="float") / 255.0


trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)

augment = ImageDataGenerator( 
            
                rotation_range=30,
                zoom_range=0.25,
                width_shift_range=0.10,
                height_shift_range=0.10,
                shear_range=0.10,
                horizontal_flip=False,
                data_format = "channels_last",
                fill_mode="nearest",
)


epochs = 15
batchsize = 6
    
    
m.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_hinge', metrics=['accuracy'])

history = m.fit(x=augment.flow(trainX, trainY, batch_size=batchsize), validation_data=(testX, testY), 
steps_per_epoch= len(trainX) // batchsize, epochs=epochs)
    


# Plot the accuracy and loss curves

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training loss')
plt.legend()

plt.show()




