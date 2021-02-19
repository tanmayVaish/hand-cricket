from capture import Capture as cp
from preprocess import Preprocess 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Capturing -----------------------------------------------------------------------------------------
data = cp.Collect()
images = preprocess.convert(data)
# ---------------------------------------------------------------------------------------------------


# Preprocessing -------------------------------------------------------------------------------------

blur_images = Preprocess.gaussian(images) # contains list of images of all classess in ascending order
seg_images = Preprocess.segmentation(blur_images)
morph_images = Preprocess.morphology(seg_images,images)

# ---------------------------------------------------------------------------------------------------

class Augument:    
    
    
    def augmentation():
        
        augment = ImageDataGenerator( 
            
                rotation_range=30,
                zoom_range=0.25,
                width_shift_range=0.10,
                height_shift_range=0.10,
                shear_range=0.10,
                horizontal_flip=False,
                fill_mode="nearest"
        )