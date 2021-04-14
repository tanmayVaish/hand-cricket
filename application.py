import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from statistics import mode, StatisticsError
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense,MaxPool2D,Dropout,Flatten,Conv2D,GlobalAveragePooling2D,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import datetime

from random import choice,shuffle
from scipy import stats as st

from collections import deque

font = cv2.FONT_HERSHEY_COMPLEX


def display_winner(winner_flag,field,cv2,font):
    if winner_flag==0:
        cv2.putText(field, "Draw", (500, 550), font, 1.2, (0,128,255), 2, cv2.LINE_AA)

    elif winner_flag==1:
        cv2.putText(field, "Winner: User", (495, 550), font, 1.2, (0,128,255), 2, cv2.LINE_AA)

    elif winner_flag==2:
        cv2.putText(field, "Winner: Computer", (455, 550), font, 1.2, (0,128,255), 2, cv2.LINE_AA)

        
def display_out(field,cv2,font):
    cv2.putText(field, "User: OUT!!", (550, 400), font, 1.2, (0,0,255), 2, cv2.LINE_AA)

#Calculate and Update score
def calculate_score(move1, move2,total_run):

    if move1 == move2:
        return "Out"
    else:       
        return str(total_run+int(move1))



CLASS_REV_MAP = {
    0: "none",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
}
    
def mapper(index):
    return CLASS_REV_MAP[index]


total_run=0
total_score="0"
user_score=0
computer_score=0
counter=0
user_out=0
computer_out=0
winner_flag=-1
startCounter = False
startTime = 0.0
timeElapsed = 0.0
nSecond = 0
totalSec = 3
computer_move="none"





model = load_model("hc.h5")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
box_size = 234
width = int(cap.get(3))


# Initially the moves will be `none`
computer_move= "none"
user_move = "none"

label_names = ['none', '1', '2', '3','4','5']

# All scores are 0 at the start.
computer_score, user_score = 0, 0

# The default color of bounding box is Blue
rect_color = (255, 0, 0)

# This variable remembers if the hand is inside the box or not.
hand_inside = False

# We will only consider predictions having confidence above this threshold.
confidence_threshold = 0.70

# Instead of working on a single prediction, we will take the mode of 5 predictions by using a deque object
# This way even if we face a false positive, we would easily ignore it
smooth_factor = 5

# Our initial deque list will have 'none' repeated 5 times.
de = deque(['none'] * 5, maxlen=smooth_factor)

while True:
    
    field = cv2.imread("field.png")
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame,1)
    
    # user's space
    cv2.rectangle(field,(300,80), (1024,768), (255,255,255), 2)
       
    # extracting user's image
    user_frame = frame[100:500, 800:1200]
    img = cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    
    img = np.array([img]).astype('float64')/255.0
    
    # Predict the move made
    pred = model.predict(img)
    
    # Get the index of the predicted class
    move_code = np.argmax(pred[0])
    
    #####################
    # all ok till here
    #####################
    
    # Get the class name of the predicted class
    user_move = label_names[move_code]
    
    # Get the confidence of the predicted class
    prob = np.max(pred[0])
    
    # Make sure the probability is above our defined threshold
    if prob >= confidence_threshold:
        
        # Now add the move to deque list from left
        de.appendleft(user_move)
        
        # Get the mode i.e. which class has occured more frequently in the last 5 moves.
        try:
            user_move = st.mode(de)[0][0] 
            
        except StatisticsError:
            print('Stats error')
            continue
        
        
        #####################
        # all ok till here
        #####################
             

        # from here i am going to make the inning system using counter variable.



        #Computer_batting 
        if counter%2 == 1:
            
            cv2.putText(field, "Computer's Batting", (465, 650), font, 1.2, (255,255,0), 2, cv2.LINE_AA)
            
            if total_score != "Out":
                if user_move != "none" and hand_inside == False:
                            
                    hand_inside = True
                    computer_move = choice(['1','2','3','4','5'])
                    
                    total_score = calculate_score(computer_move,user_move,total_run)
                    
                    if total_score!="Out":
                        computer_score=int(total_score)
                
                elif user_move == "none":
                    
                    hand_inside = False
                
                
                else:
                    computer_move = "none"
            else:
                computer_out=1
                total_score="0"
                
                
            if computer_move != "none":
                computer_emoji = cv2.imread("icons/{}.png".format(computer_move))
                print(computer_emoji)
                computer_emoji = cv2.resize(computer_emoji,(400,400))
                field[100:500, 100:500] = computer_emoji
                    
                


            #increasing score
            if total_score!="Out":
                total_run=int(total_score)
        
            # displaying the information
            cv2.putText(field,"User Move: " + user_move, (840, 50), font, 1.2, (0, 255, 255),2,cv2.LINE_AA)
            cv2.putText(field, "Computer Move: " + computer_move, (65, 50), font, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(field, "Computer score: " + str(computer_score), (70, 600), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(field, "User score: " + str(user_score), (840, 600), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        
    
            
    
            #change of innings
            if computer_out or user_score<computer_score:
                if user_score>computer_score:    
                    winner_flag=1
                    print("WINNER USER")
                if user_score==computer_score:    
                    winner_flag=0           
                    print("DRAW")
                if user_score<computer_score:    
                    winner_flag=2
                    print("WINNER COMPUTER")    
        
                startCounter = True
                startTime = datetime.datetime.now()
                counter=counter+1
                computer_out=0
    
    



        #User_batting 
        if counter%2 == 0:
            
            cv2.putText(field, "User's Batting", (465, 650), font, 1.2, (255,255,0), 2, cv2.LINE_AA)
            
            if total_score != "Out":
                if user_move != "none" and hand_inside == False:
                            
                    hand_inside = True
                    computer_move = choice(['1','2','3','4','5'])
                    
                    total_score = calculate_score(user_move,computer_move,total_run)
                    
                    if total_score!="Out":
                        user_score=int(total_score)
                
                elif user_move == "none":
                    hand_inside = False

                
                
                else:
                    computer_move = "none"
            else:
                user_out=1
                startTime = datetime.datetime.now()  
                total_score="0"
                
                
            if computer_move != "none":
                computer_emoji = cv2.imread("icons/{}.png".format(computer_move))
                computer_emoji = cv2.resize(computer_emoji,(400,400))
                field[100:500, 100:500] = computer_emoji
                    
            #increasing score
            if total_score!="Out":
                total_run=int(total_score)   


            # displaying the information
            cv2.putText(field,"User Move: " + user_move, (840, 50), font, 1.2, (0, 255, 255),2,cv2.LINE_AA)
            cv2.putText(field, "Computer Move: " + computer_move, (65, 50), font, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(field, "Computer score: " + str(computer_score), (70, 600), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(field, "User score: " + str(user_score), (840, 600), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    



            #change of innings
            if user_out:
                
                
                if nSecond < totalSec:
                    
                    display_out(field,cv2,font)
    #                cv2.putText(field, "User: OUT!!", (500, 250), font, 1.2, (0,128,255), 2, cv2.LINE_AA)
    
                    timeElapsed = (datetime.datetime.now() - startTime).total_seconds()
            
                    if timeElapsed >= 1:
                        nSecond += 1
                        timeElapsed = 0
                        startTime = datetime.datetime.now()
                else:
                    startTime = 0.0
                    timeElapsed = 0.0
                    nSecond = 1
                    counter=counter+1
                    user_out=0
                
        if startCounter:
    #        nSecond=0
            if nSecond < totalSec:
                display_winner(winner_flag,field,cv2,font)
                timeElapsed = (datetime.datetime.now() - startTime).total_seconds()
                
                if timeElapsed >= 1:
                    nSecond += 1
                    timeElapsed = 0
                    startTime = datetime.datetime.now()
            else:
                startCounter = False
                nSecond = 1
                startTime = 0.0
                timeElapsed = 0.0
                nSecond = 1
                winner_flag = -1
                user_score=0
                computer_score=0
                
    field[100:500, 800:1200]=user_frame
   
    cv2.putText(field, "Take out your hand from the box for next ball" , (200, 700), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Handy cricket", field)
    
    #To end the game press 'q'
    k = cv2.waitKey(10)
    if k == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()