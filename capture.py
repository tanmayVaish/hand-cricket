import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
import os


class Capture:
        
    def Collect(num_samples):
        
        global one,two,three,four,five,none
        capture = cv.VideoCapture(0)
    
        count = 0
        switch = False
        
        # This the ROI size, the size of images saved will be box_size -10
        box_size = 234
        
        # Getting the width of the frame from the camera properties
        width = int(capture.get(3))
        
        while True:
            
            # Read frame by frame
            ret, frame = capture.read()
            
            # Flip the frame laterally
            frame = cv.flip(frame, 1)
            
            # Break the loop if there is trouble reading the frame.
            if not ret:
                break
                
                
                
            # If counter is equal to the number samples then reset triger and the counter
            if count == num_samples:
                switch = not switch
                count = 0
            
            
            cv.rectangle(frame, (width - box_size, 0), (width, box_size), (0, 0, 0), 2)
            
            cv.namedWindow("Collecting images", cv.WINDOW_NORMAL) 
            
            
        
            if switch == True:
                
                ROI = frame[5: box_size-5 , width-box_size+5: width-5] # LFU
                eval(class_name).append([ROI])
                    
                # Increment the counter 
                count += 1 
            
                # Text for the counter
                text = "Collected Samples of {}: {}".format(class_name, count)
                
            else:
                text = "Press 1-0 and n, for collecting data-set."
            
            # Show the counter on the imaege
            cv.putText(frame, text, (3, 350), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv.LINE_AA) 
            
            # Display the window
            cv.imshow("Collecting images", frame)
            
            
            
            # Wait 1 ms
            k = cv.waitKey(1)
            
                 
            if k == ord('1'):
                switch = not switch
                class_name = "one" # this is defined for eval(): This will represent the one[]
                one = []
                
            if k == ord('2'):
                switch = not switch
                class_name = "two"
                two = []
                
            if k == ord('3'):
                switch = not switch
                class_name = "three"
                three = []
                
            if k == ord('4'):
                switch = not switch
                class_name = "four"
                four = []
            
            if k == ord('5'):
                switch = not switch
                class_name = "five"
                five = []
            if k == ord('n'):
                switch = not switch
                class_name = "none"
                none = []
                    
            if k == ord('q'):
                break
                
                
        #  Release the camera and destroy the window
        capture.release()
        cv.destroyAllWindows()
        
        data = [one,two,three,four,five,none]
        
        # return all the lists containing our dataset!
        return data
    

    
    
    def Show(data):
        
        # Set the figure size
        plt.figure(figsize=[30,20])
        
        # Set the rows and columns
        rows, cols = 6,10
        
        # Iterate for each class
        for class_index, class_name in enumerate(data):
            
            r = np.random.randint(10, size=8);
            
            for i, example_index in enumerate(r,1):
                plt.subplot(rows,cols,class_index*cols + i);
                plt.imshow(class_name[example_index][0][:,:,::-1]); # converting BGR to RGB
                plt.axis('off');
                