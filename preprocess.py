from capture import Capture as cp

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#data = cp.Collect()
#one,two,three,four,five,none = data

# data = [one,two,three,four,five,none] list of classes of images 
# images = [1,2,3,4,5,6,7,8.....,60] list of images!!


class Preprocess:        

    def convert(data):
        
        images = []
        for classes in data:
            for i in classes:
                for j in i:
                    images.append(j)
        return images
    

    def gaussian(images):
        
        lst = []
        
        for i in images:
            blur = cv.GaussianBlur(i,(5,5),0)
            lst.append(blur)
        return lst


    def segmentation(images):
        
        lst = []

        for i in images:
            gray = cv.cvtColor(i, cv.COLOR_RGB2GRAY)
            ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            lst.append(thresh)
        return lst            



    def morphology(seg_images,images):

        lst = []
        
        for i in range(len(seg_images)):
            
            # Further noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv.morphologyEx(seg_images[i], cv.MORPH_OPEN, kernel, iterations=2)
            sure_bg = cv.dilate(opening, kernel, iterations=3)
    
            # Finding sure foreground area
            dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
            ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv.subtract(sure_bg, sure_fg)
    
            # Marker labelling
            ret, markers = cv.connectedComponents(sure_fg)
    
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1
    
            # Now, mark the region of unknown with zero
            markers[unknown == 255] = 0
    
            markers = cv.watershed(images[i], markers)
            images[i][markers == -1] = [255, 0, 0]
            lst.append(sure_fg)
            
        return lst



def Show(images):

    plt.figure(figsize=[60,50])
    columns = 10
    rows = 6
    for i in range(1, columns*rows ):
        plt.subplot(rows, columns, i)
        plt.imshow(images[i][:,::-1])
        plt.axis('off')





if __name__ == "__main__":

    images = Preprocess.convert(data)
    
    blur_images = Preprocess.gaussian(images) # contains list of images of all classess in ascending order
    seg_images = Preprocess.segmentation(images)
    morph_images = Preprocess.morphology(seg_images,images)
    
    Show(morph_images)
    