from tensorflow.keras.models import Model, load_model
import cv2
import numpy as np
from random import choice
from collections import deque
from scipy import stats as st

model = load_model("hc.h5")
run_dict = {"none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

def findout_winner(move1, move2):
    if move1 == move2:
        return 0
    else:
        return 1
    

def display_computer_move(computer_move_name, frame):
    icon = cv2.imread( "icons/{}.png".format(computer_move_name), 1)
    icon = cv2.resize(icon, (224,224))
    # This is the portion which we are going to replace with the icon image
    roi = frame[0:224, 0:224]
    # Get binary mask from the transparent image, 4th channel is the alpha channel
    mask = icon[:,:,-1]
    # Making the mask completely binary (black & white)
    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
    # Store the normal bgr image
    icon_bgr = icon[:,:,:3]
    # Now combine the foreground of the icon with background of ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask = cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(icon_bgr, icon_bgr, mask = mask)
    combined = cv2.add(img1_bg, img2_fg)
    frame[0:224, 0:224] = combined
    return frame
    
def show_winner(user_score, computer_score):    
    if user_score > computer_score:
        img = cv2.imread("images/youwin.jpg")   
    elif user_score < computer_score:
        img = cv2.imread("images/comwins.jpg")    
    else:
        img = cv2.imread("images/draw.jpg")      
    #cv2.putText(img, "Press 'ENTER' to play again, else exit",
                #(150, 530), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)     
    cv2.imshow("Rock Paper Scissors", img)     
    # If enter is pressed.
    k = cv2.waitKey(0)     
    # If the user presses 'ENTER' key then return TRUE, otherwise FALSE
    if k == 13:
        return True 
    else:
        return False
    
cap = cv2.VideoCapture(0)
box_size = 234
width = int(cap.get(3))
# Initially the moves will be `nothing`
computer_move_name= "none"
final_user_move = "none"
label_names = ['none', '1', '2', '3', '4', '5']
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
# Our initial deque list will have 'nothing' repeated 5 times.
de = deque(['none'] * 5, maxlen=smooth_factor)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    cv2.namedWindow("Hand Cricket", cv2.WINDOW_NORMAL)
    # extract the region of image within the user rectangle
    roi = frame[5: box_size-5 , width-box_size + 5: width -5]
    roi = np.array([roi]).astype('float64') / 255.0
    # Predict the move made
    pred = model.predict(roi)
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

            final_user_move = st.mode(de)[0][0]
        except StatisticsError:
            print('Stats error')
            continue        
        # If nothing is not true and hand_inside is False then proceed.
        # Basically the hand_inside variable is helping us to not repeatedly predict during the loop
        # So now the user has to take his hands out of the box for every new prediction.
        if final_user_move != "none" and hand_inside == False:
            # Set hand inside to True
            hand_inside = True
            # Get Computer's move and then get the winner.
            computer_move_name = choice(['1', '2', '3', '4', '5'])
            winner = findout_winner(final_user_move, computer_move_name)
            # Display the computer's move
            display_computer_move(computer_move_name, frame)
            # If winner is computer then it gets points and vice versa.
            # We're also changing the color of rectangle based on who wins the round.
            if winner == 1:
                user_score += int(final_user_move)
                rect_color = (0, 0, 255)
            #elif winner == 0:
            #    user_score += 1;
            #    rect_color = (0, 250, 0)
            # winner == "Tie":
            #    rect_color = (255, 250, 255)
            # If all the attempts are up then find our the winner     
            if winner == 0:
                play_again = show_winner(user_score, computer_score)
                # If the user pressed Enter then restart the game by re initializing all variables
                if play_again:
                    user_score, computer_score = 0, 0
                # Otherwise quit the program.
                else:
                    break
        # Display images when the hand is inside the box even when hand_inside variable is True.
        elif final_user_move != "nothing" and hand_inside == True:
            display_computer_move(computer_move_name, frame)
        # If class is nothing then hand_inside becomes False
        elif final_user_move == 'nothing':           
            hand_inside = False
            rect_color = (255, 0, 0)
    # This is where all annotation is happening.
    cv2.putText(frame, "Your Move: " + final_user_move,

                    (420, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (2, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Your Score: " + str(user_score),
                    (420, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Computer Score: " + str(computer_score),
                    (2, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (width - box_size, 0), (width, box_size), rect_color, 2)
    # Display the image   
    cv2.imshow("Hand Cricket", frame)
    # Exit if 'q' is pressed
    k = cv2.waitKey(10)
    if k == ord('q'):
        break
# Relase the camera and destroy all windows.
cap.release()
cv2.destroyAllWindows()