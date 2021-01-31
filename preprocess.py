from capture import Capture as cp

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = cp.Collect()
one,two,three,four,five,none = data




class preprocess:

    def augment():
        
        # Adding transformations that I know would help, you can feel free to add more.
        # I'm doing horizontal_flip = False, in case you aren't sure which hand you would be using you can make that True.
        
        augment = ImageDataGenerator( 
                rotation_range=30,
                zoom_range=0.25,
                width_shift_range=0.10,
                height_shift_range=0.10,
                shear_range=0.10,
                horizontal_flip=False,
                fill_mode="nearest")
        
        
    def gaussian():

        images = []
        for classes in data:
            for i in classes:
                for j in i:

                    img = j
                    blur = cv.GaussianBlur(img,(5,5),0)
                    images.append(blur)

        return images



    def morphology(thresh,images):

        # Further noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)

        #Displaying segmented back ground
        preprocess.display(images[0], sure_bg, 'Original', 'Segmented Background')


        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        markers = cv.watershed(images[0], markers)
        images[0][markers == -1] = [255, 0, 0]

        # Displaying markers on the image
        preprocess.display(images[0], markers, 'Original', 'Marked')




    def Show(images):

        plt.figure(figsize=[40, 30])
        columns = 10
        rows = 6
        for i in range(1, columns*rows ):
            plt.subplot(rows, columns, i)
            plt.imshow(images[i][:,:,::-1])
            plt.axis('off')








    def segmentation(images):

        gray1 = cv.cvtColor(images[0], cv.COLOR_RGB2GRAY)
        gray2 = cv.cvtColor(images[11], cv.COLOR_RGB2GRAY)
        gray3 = cv.cvtColor(images[55], cv.COLOR_RGB2GRAY)

        ret1, thresh1 = cv.threshold(gray1, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        ret2, thresh2 = cv.threshold(gray2, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        ret3, thresh3 = cv.threshold(gray3, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # Displaying segmented images
        preprocess.display(images[0], thresh1, 'Original', 'Segmented')
        preprocess.display(images[11], thresh2, 'Original', 'Segmented')
        preprocess.display(images[55], thresh3, 'Original', 'Segmented')

        return thresh1










    def display(a, b, title1 = "Original", title2 = "Edited"):
        plt.subplot(121), plt.imshow(a), plt.title(title1)
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(b), plt.title(title2)
        plt.xticks([]), plt.yticks([])
        plt.show()












if __name__ == "__main__":

    images = preprocess.gaussian() # contains list of images of all classess in ascending order

    preprocess.Show(images) # Blurred
    cp.Show(data) # original
    thresh = preprocess.segmentation(images)
    preprocess.morphology(thresh, images)
