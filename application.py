import cv2
import numpy as np
from statistics import StatisticsError
from tensorflow.keras.models import load_model

import datetime

from random import choice
from scipy import stats as st

from collections import deque

font = cv2.FONT_HERSHEY_DUPLEX

#Display the winner
def display_winner(winner_flag,stadium,cv2,font):
    if winner_flag==0:
        cv2.putText(stadium, "Draw", (500, 550), font, 1.2, (0,128,255), 2, cv2.LINE_AA)

    elif winner_flag==1:
        cv2.putText(stadium, "Winner: User", (495, 550), font, 1.2, (0,128,255), 2, cv2.LINE_AA)

    elif winner_flag==2:
        cv2.putText(stadium, "Winner: Computer", (455, 550), font, 1.2, (0,128,255), 2, cv2.LINE_AA)


def display_out(stadium,cv2,font):
    cv2.putText(stadium, "User: OUT!!", (550, 400), font, 1.2, (0,0,255), 2, cv2.LINE_AA)

#Calculate and Update score
def calculate_score(move1, move2,total_runs):
    if move1 == move2:
        return "Out"
    else:       
        return str(total_runs+int(move1))

run_dict = {
    0: "none",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
}
    
def mapper(index):
    return run_dict[index]

total_runs=0
total_score="0"
user_score=0
cmp_score=0
flag=0
user_out=0
cmp_out=0
winner_flag=-1
startCounter = False
startTime = 0.0
timeElapsed = 0.0
nSecond = 0
totalSec = 3
cmp_move="none"





model = load_model("hc.h5")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
box_size = 234
width = int(cap.get(3))


# Initially the moves will be `none`
cmp_move= "none"
user_move = "none"

label_names = ['none', '1', '2', '3','4','5']

# All scores are 0 at the start.
cmp_score, user_score = 0, 0

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
    
    stadium = cv2.imread("stadium.jpg")
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame,1)
    
    # user's space
    cv2.rectangle(stadium,(300,80), (1024,768), (255,255,255), 2)
       
    # extracting user's image
    user_frame = frame[100:500, 800:1200]
    img = cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    
    img = np.array([img]).astype('float64')/255.0
    
    # Predict the move made
    pred = model.predict(img)
    
    # Get the index of the predicted class
    move_code = np.argmax(pred[0])

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

        #The Inning system works using the flag variable.
        #Computer_batting 
        if flag%2 == 1:
            
            cv2.putText(stadium, "Computer's Batting", (465, 650), font, 1.2, (255,255,0), 2, cv2.LINE_AA)
            
            if total_score != "Out":
                if user_move != "none" and hand_inside == False:
                            
                    hand_inside = True
                    cmp_move = choice(['1','2','3','4','5'])
                    
                    total_score = calculate_score(cmp_move,user_move,total_runs)
                    
                    if total_score!="Out":
                        cmp_score=int(total_score)
                
                elif user_move == "none":
                    
                    hand_inside = False
                
                
                else:
                    cmp_move = "none"
            else:
                cmp_out=1
                total_score="0"
                
                
            if cmp_move != "none":
                cmp_emoji = cv2.imread("icons/{}.jpg".format(cmp_move))
                cmp_emoji = cv2.resize(cmp_emoji,(400,400))
                stadium[100:500, 100:500] = cmp_emoji
                    
                


            #increasing score
            if total_score!="Out":
                total_runs=int(total_score)
        
            # displaying the information
            cv2.putText(stadium,"User Move: " + user_move, (840, 50), font, 1.2, (0, 255, 255),2,cv2.LINE_AA)
            cv2.putText(stadium, "Computer Move: " + cmp_move, (65, 50), font, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(stadium, "Computer score: " + str(cmp_score), (70, 600), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(stadium, "User score: " + str(user_score), (840, 600), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        
    
            
    
            #change of innings
            if cmp_out or user_score<cmp_score:
                if user_score>cmp_score:    
                    winner_flag=1
                    print("WINNER USER")
                if user_score==cmp_score:    
                    winner_flag=0           
                    print("DRAW")
                if user_score<cmp_score:    
                    winner_flag=2
                    print("WINNER COMPUTER")    
        
                startCounter = True
                startTime = datetime.datetime.now()
                flag=flag+1
                cmp_out=0
    
    



        #User_batting 
        if flag%2 == 0:
            
            cv2.putText(stadium, "User's Batting", (465, 650), font, 1.2, (255,255,0), 2, cv2.LINE_AA)
            
            if total_score != "Out":
                if user_move != "none" and hand_inside == False:
                            
                    hand_inside = True
                    cmp_move = choice(['1','2','3','4','5'])
                    
                    total_score = calculate_score(user_move,cmp_move,total_runs)
                    
                    if total_score!="Out":
                        user_score=int(total_score)
                
                elif user_move == "none":
                    hand_inside = False

                
                
                else:
                    cmp_move = "none"
            else:
                user_out=1
                startTime = datetime.datetime.now()  
                total_score="0"
                
                
            if cmp_move != "none":
                cmp_emoji = cv2.imread("icons/{}.jpg".format(cmp_move))
                cmp_emoji = cv2.resize(cmp_emoji,(400,400))
                stadium[100:500, 100:500] = cmp_emoji
                    
            #increasing score
            if total_score!="Out":
                total_runs=int(total_score)   


            # displaying the information
            cv2.putText(stadium,"User Move: " + user_move, (840, 50), font, 1.2, (0, 255, 255),2,cv2.LINE_AA)
            cv2.putText(stadium, "Computer Move: " + cmp_move, (65, 50), font, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(stadium, "Computer score: " + str(cmp_score), (70, 600), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(stadium, "User score: " + str(user_score), (840, 600), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    



            #change of innings
            if user_out:
                
                
                if nSecond < totalSec:
                    
                    display_out(stadium,cv2,font)
    #                cv2.putText(stadium, "User: OUT!!", (500, 250), font, 1.2, (0,128,255), 2, cv2.LINE_AA)
    
                    timeElapsed = (datetime.datetime.now() - startTime).total_seconds()
            
                    if timeElapsed >= 1:
                        nSecond += 1
                        timeElapsed = 0
                        startTime = datetime.datetime.now()
                else:
                    startTime = 0.0
                    timeElapsed = 0.0
                    nSecond = 1
                    flag=flag+1
                    user_out=0
                
        if startCounter:
    #        nSecond=0
            if nSecond < totalSec:
                display_winner(winner_flag,stadium,cv2,font)
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
                cmp_score=0
                
    stadium[100:500, 800:1200]=user_frame
   
    cv2.putText(stadium, "Take out your hand from the box for next ball" , (200, 700), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Handy cricket", stadium)
    
    #To end the game press 'q'
    k = cv2.waitKey(10)
    if k == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()