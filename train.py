from model import Model 
from capture import Capture as cp
from preprocess import Preprocess 
from sklearn.model_selection import train_test_split


# Capturing -----------------------------------------------------------------------------------------
num_samples = 10
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