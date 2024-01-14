import tkinter as tk
from tkinter import *
import tkinter.messagebox as tmsg
from tkinter import filedialog
from tkinter.filedialog import askopenfilename, askopenfile
from tkVideoPlayer import TkinterVideo
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
import cv2
import os

from tkVideoPlayer import TkinterVideo
from tkvideo import tkvideo


###################################################################################################################################
# Proposed Model's H5 file on JHMDB Datasets

model_1 = load_model("model/JHMDB.h5/")
predicted_class_name = ''

image_height , image_width = 96,96


# Specify the number of frames of a video that will be fed to the model as one sequence.
window_size = 25


# Specify the directory containing the JHMDB dataset.
#DATASET_DIR = "D:\\aman_m_jhmdb_all"     # in colab :-- "/content/drive/MyDrive/JHMDB_video/ReCompress_Videos"


# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
#classes_list = ['Kicking-Side', 'Swing-SideAngle', 'SkateBoarding-Front', 'Walk-Front', 'Golf-Swing-Side', 'Swing-Bench', 'Run-Side', 'Kicking-Front', 'Lifting', 'Riding-Horse', 'Golf-Swing-Front', 'Golf-Swing-Back', 'Diving side']
classes_list_1 = ['shoot_bow', 'shoot_gun', 'walk', 'stand', 'wave', 'sit', 'swing_baseball', 'throw', 'shoot_ball', 'pour', 'climb_stairs', 'jump', 'pick', 'run', 'push', 'pullup', 'golf', 'kick_ball', 'clap', 'brush_hair', 'catch']

# creating a function for Prediction
def predict_on_live_video_1(video_file_path, window_size):
    # predicted_class_name = ''

    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen = window_size)

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)

    # Getting the width and height of the video
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object


    while True:

        # Reading The Frame
        status, frame = video_reader.read()

        if not status:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model_1.predict(np.expand_dims(normalized_frame, axis = 0), verbose = 0)[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:

            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)

            # Calculating Average of Predicted Labels Probabilities Column Wise
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)

            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list_1[predicted_label]
            video_reader.release()
            return predicted_class_name


##########
# Proposed Model's H5 file on KARD Dataset
model2 = load_model("model/KARD_Video_Model.h5/")
predicted_class_name = ''

image_height , image_width = 96,96


# Specify the number of frames of a video that will be fed to the model as one sequence.
window_size = 25


# Specify the directory containing the JHMDB dataset.
#DATASET_DIR = "D:\\aman_m_jhmdb_all"     # in colab :-- "/content/drive/MyDrive/JHMDB_video/ReCompress_Videos"


# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
#classes_list = ['Kicking-Side', 'Swing-SideAngle', 'SkateBoarding-Front', 'Walk-Front', 'Golf-Swing-Side', 'Swing-Bench', 'Run-Side', 'Kicking-Front', 'Lifting', 'Riding-Horse', 'Golf-Swing-Front', 'Golf-Swing-Back', 'Diving side']
classes_list_2 = ['Take Umbrella', 'Sit down', 'Drink', 'Side Kick', 'Bend', 'Catch Cap', 'Toss Paper', 'Stand up', 'High arm wave', 'Horizontal arm wave', 'Walk', 'Hand Clap', 'Draw X', 'Forward Kick', 'Two hand wave', 'High throw', 'Phone Call', 'Draw Tick']
# creating a function for Prediction
def predict_on_live_video_2(video_file_path, window_size):
    # predicted_class_name = ''

    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen = window_size)

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)

    # Getting the width and height of the video
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object


    while True:

        # Reading The Frame
        status, frame = video_reader.read()

        if not status:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model2.predict(np.expand_dims(normalized_frame, axis = 0), verbose = 0)[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:

            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)

            # Calculating Average of Predicted Labels Probabilities Column Wise
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)

            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list_2[predicted_label]
            video_reader.release()
            return predicted_class_name

#######
# Proposed Model's H5 file on UCF_Sports

model3 = load_model("model/UCF_Sports_proposed_model_new.h5")
predicted_class_name = ''

image_height , image_width = 96,96


# Specify the number of frames of a video that will be fed to the model as one sequence.
window_size = 25


# Specify the directory containing the JHMDB dataset.
#DATASET_DIR = "D:\\aman_m_jhmdb_all"     # in colab :-- "/content/drive/MyDrive/JHMDB_video/ReCompress_Videos"


# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
#classes_list = ['Kicking-Side', 'Swing-SideAngle', 'SkateBoarding-Front', 'Walk-Front', 'Golf-Swing-Side', 'Swing-Bench', 'Run-Side', 'Kicking-Front', 'Lifting', 'Riding-Horse', 'Golf-Swing-Front', 'Golf-Swing-Back', 'Diving side']
classes_list_3 = ['Kicking-Front', 'Run-Side', 'Golf-Swing-Side', 'Swing-Bench', 'Swing-SideAngle', 'Kicking-Side', 'Riding-Horse', 'Walk-Front', 'SkateBoarding-Front', 'Lifting', 'Golf-Swing-Front', 'Diving side', 'Golf-Swing-Back']

# creating a function for Prediction
def predict_on_live_video_3(video_file_path, window_size):
    # predicted_class_name = ''

    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen = window_size)

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)

    # Getting the width and height of the video
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object


    while True:

        # Reading The Frame
        status, frame = video_reader.read()

        if not status:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model3.predict(np.expand_dims(normalized_frame, axis = 0), verbose = 0)[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:

            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)

            # Calculating Average of Predicted Labels Probabilities Column Wise
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)

            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list_3[predicted_label]
            video_reader.release()
            return predicted_class_name

########
# Proposed Model's H5 file on UT_Interaction
model4 = load_model("model/UT-Intersection_Model.h5")
predicted_class_name = ''

image_height , image_width = 96,96


# Specify the number of frames of a video that will be fed to the model as one sequence.
window_size = 25


# Specify the directory containing the JHMDB dataset.
#DATASET_DIR = "D:\\aman_m_jhmdb_all"     # in colab :-- "/content/drive/MyDrive/JHMDB_video/ReCompress_Videos"


# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
#classes_list = ['Kicking-Side', 'Swing-SideAngle', 'SkateBoarding-Front', 'Walk-Front', 'Golf-Swing-Side', 'Swing-Bench', 'Run-Side', 'Kicking-Front', 'Lifting', 'Riding-Horse', 'Golf-Swing-Front', 'Golf-Swing-Back', 'Diving side']
classes_list_4 = ['hand shake', 'hugging', 'kick', 'pointting', 'punching', 'push']
# creating a function for Prediction
def predict_on_live_video_4(video_file_path, window_size):
    # predicted_class_name = ''

    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen = window_size)

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)

    # Getting the width and height of the video
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object


    while True:

        # Reading The Frame
        status, frame = video_reader.read()

        if not status:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model4.predict(np.expand_dims(normalized_frame, axis = 0), verbose = 0)[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:

            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)

            # Calculating Average of Predicted Labels Probabilities Column Wise
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)

            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list_4[predicted_label]
            video_reader.release()
            return predicted_class_name
        
#######

# Proposed Model's H5 file on UCF50 Dataset
model4 = load_model("model/UCF50_Model.h5/")
predicted_class_name = ''

image_height , image_width = 96,96


# Specify the number of frames of a video that will be fed to the model as one sequence.
window_size = 25


# Specify the directory containing the JHMDB dataset.
#DATASET_DIR = "D:\\aman_m_jhmdb_all"     # in colab :-- "/content/drive/MyDrive/JHMDB_video/ReCompress_Videos"


# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
#classes_list = ['Kicking-Side', 'Swing-SideAngle', 'SkateBoarding-Front', 'Walk-Front', 'Golf-Swing-Side', 'Swing-Bench', 'Run-Side', 'Kicking-Front', 'Lifting', 'Riding-Horse', 'Golf-Swing-Front', 'Golf-Swing-Back', 'Diving side']
classes_list_5 = ['BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billiards', 'BreastStroke', 'CleanAndJerk', 'Diving', 'Drumming', 'Fencing', 'GolfSwing', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Lunges', 'MilitaryParade', 'Mixing', 'Nunchucks', 'PizzaTossing', 'PlayingGuitar', 'PlayingPiano', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Swing', 'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog', 'YoYo']
# creating a function for Prediction
def predict_on_live_video_5(video_file_path, window_size):
    # predicted_class_name = ''

    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen = window_size)

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)

    # Getting the width and height of the video
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object


    while True:

        # Reading The Frame
        status, frame = video_reader.read()

        if not status:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model4.predict(np.expand_dims(normalized_frame, axis = 0), verbose = 0)[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:

            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)

            # Calculating Average of Predicted Labels Probabilities Column Wise
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)

            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list_5[predicted_label]
            video_reader.release()
            return predicted_class_name

# model task end

########
        
# GUI making started 

root =Tk()

root.title("Video Class Predictions by Aparna")
root.geometry("1000x1280")
# set the dimensions of the window to full screen
root.attributes('-fullscreen', True)

# create a custom title bar
title_bar = Frame(root, bg='white', relief='raised', bd=2, height=30)
title_bar.place(relx=0, rely=0, relwidth=1, height=25)

# create a label for the title bar
title_label = Label(title_bar, text="Human Activity Recoginition", bg='white', fg='black', font=('Arial', 10))
title_label.pack(side='left', padx=2)

# create a close button
close_button = Button(title_bar, text='X', command=root.destroy)
close_button.pack(side='right', padx=2)

# create a maximize button
maximize_button = Button(title_bar, text='[]', command=lambda: root.attributes('-zoomed', True))
maximize_button.pack(side='right', padx=2)

# create a minimize button
minimize_button = Button(title_bar, text='-', command=lambda: root.iconify())
minimize_button.pack(side='right', padx=2)

# enable the minimize and close buttons
root.protocol("WM_DELETE_WINDOW", root.destroy)
root.protocol("WM_MINIMIZE_WINDOW", lambda: root.attributes('-fullscreen', False))

#set background color
root.configure(bg='sky blue')

file = ''
text = ''

# to fetch photos from normal space
def filename():
    t1.config(state='normal')
    t1.delete('1.0', END)
    file = askopenfilename()
    # return filename
    t1.insert(tk.END, str(file))
    t1.config(state='disabled')
    # print(filename)


def getfile1():
    t2.config(state='normal')
    t2.delete('1.0', END)
    text = t1.get(1.0, "end-1c")
    # file = askopenfilename()
    # return filename
    p_class = predict_on_live_video_1(text, window_size)
    t2.insert(tk.END, str(p_class))
    t2.config(state='disabled')

def getfile2():
    t2.config(state='normal')
    t2.delete('1.0', END)
    text = t1.get(1.0, "end-1c")
    # file = askopenfilename()
    # return filename
    p_class = predict_on_live_video_2(text, window_size)
    t2.insert(tk.END, str(p_class))
    t2.config(state='disabled')

def getfile3():
    t2.config(state='normal')
    t2.delete('1.0', END)
    text = t1.get(1.0, "end-1c")
    # file = askopenfilename()
    # return filename
    p_class = predict_on_live_video_3(text, window_size)
    t2.insert(tk.END, str(p_class))
    t2.config(state='disabled')

def getfile4():
    t2.config(state='normal')
    t2.delete('1.0', END)
    text = t1.get(1.0, "end-1c")
    # file = askopenfilename()
    # return filename
    p_class = predict_on_live_video_4(text, window_size)
    t2.insert(tk.END, str(p_class))
    t2.config(state='disabled')
    
def getfile5():
    t2.config(state='normal')
    t2.delete('1.0', END)
    text = t1.get(1.0, "end-1c")
    # file = askopenfilename()
    # return filename
    p_class = predict_on_live_video_5(text, window_size)
    t2.insert(tk.END, str(p_class))
    t2.config(state='disabled')

"""def get_accuracy():
    #tmsg.showinfo("Accuracy","Accuracy on this Video Classifier using pre-trained model is 99.94%")
    a = print("Accuacy on this Video/Image is :99.94%")"""



def aman(event):
    print(f"you have clicked on button at {event.x},{event.y}")

def open_file():
    file = askopenfile(mode='r', filetypes=[
        ('Video Files', ["*.avi","*.mp4"])])
    if file is not None:
        global filename

        filename = file.name
        global videoplayer
        videoplayer = TkinterVideo(root, scaled=True)
        videoplayer.load(r"{}".format(filename))
        # videoplayer.config(width=50, height=650)
        videoplayer.place(x=650, y=550, width=650, height=400)
        videoplayer.play()
        #add_text_to_video()

"""# Define the video player functions
def open_file():
    file = filedialog.askopenfile(mode='r', filetypes=[('Video Files', ["*.avi"])])
    if file is not None:
        global filename
        filename = file.name
        global videoplayer
        videoplayer = TkinterVideo(master=root, scaled=True)
        videoplayer.load(r"{}".format(filename))
        videoplayer.place(x=700, y=600, width=400, height=300, anchor=CENTER)
        videoplayer.play()
        add_text_to_video()"""


# def add_text_to_video():
#     cap = cv2.VideoCapture(filename)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     color = (0, 0, 255)
#     thickness = 2
#
#     # loop through each class and add text to video
#     # classes = ["Class A", "Class B", "Class C"]
#     classes_list = ['shoot_bow', 'shoot_gun', 'walk', 'stand', 'wave', 'sit', 'swing_baseball', 'throw', 'shoot_ball',
#                     'pour', 'climb_stairs', 'jump', 'pick', 'run', 'push', 'pullup', 'golf', 'kick_ball', 'clap',
#                     'brush_hair', 'catch']
#
#     for class_name in classes_list:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             height, width, channels = frame.shape
#             textsize = cv2.getTextSize(class_name, font, 1, thickness)
#             x = 10 #int((width - textsize[0]) / 2)
#             y = 10 #int((height + textsize[1]) / 2)
#             cv2.putText(frame, class_name, (x, y), font, 1, color, thickness, cv2.LINE_AA)
#             cv2.imshow('Video', frame)
#             if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
#                 break
#
#     cap.release()
#     cv2.destroyAllWindows()

def playAgain():
    print(filename)
    videoplayer.play()


def StopVideo():
    print(filename)
    videoplayer.stop()
    
def PauseVideo():
    print(filename)
    videoplayer.pause()    

"""def get_accuracy():
    print("The accuracy of class is 99")"""

# widget = Button(root,text = "Click Here")
# widget.pack(fill=BOTH)

# widget.bind("<Button-1>",aman)
# widget.bind("<Double-Button-1>",quit)

l1 = Label(root ,text='Video Based Human Activity Recognition' ,font="Georgia 21 "  ,fg='black')
l1.place(x= 700, y = 70)


button1 = Button(root,text='File Extractor' ,padx=12 ,pady=23 ,bg='brown' ,command=filename, width=7, height=1)
button1.pack(side=LEFT)



button2 = Button(root ,text='JHMDB\nClassifier' ,padx=12 ,pady=23 ,bg='brown' ,command=getfile1, width=6, height=1)
button2.pack(side=RIGHT)

button3 = Button(root ,text='KARD\nClassifier' ,padx=12 ,pady=23 ,bg='brown' ,command=getfile2, width=6, height=1)
button3.pack(side=RIGHT)

button4 = Button(root ,text='UCF_Sports\nClassifier' ,padx=12 ,pady=23 ,bg='brown' ,command=getfile3, width=6, height=1)
button4.pack(side=RIGHT)

button5 = Button(root ,text='Ut-Interaction\nClassifier' ,padx=12 ,pady=23 ,bg='brown' ,command=getfile4, width=7, height=1)
button5.pack(side=RIGHT)

button6 = Button(root ,text='UCF-50\nClassifier' ,padx=12 ,pady=23 ,bg='brown' ,command=getfile5, width=6, height=1)
button6.pack(side=RIGHT)

"""buttonnew = Button(root ,text='Display_Accuracy' ,padx=12 ,pady=23 ,bg='brown' ,command=get_accuracy)
buttonnew.pack(side=RIGHT)"""




button6 = Button(root,text='Open' ,padx=12 ,pady=23 ,bg='brown',command=open_file, width=6, height=1)
button6.pack(side=LEFT)

button7 = Button(root,text='Play' ,padx=12 ,pady=23 ,bg='brown',command=playAgain, width=6, height=1)
button7.pack(side=LEFT)

button8 = Button(root,text='Stop' ,padx=12 ,pady=23 ,bg='brown',command=StopVideo, width=6, height=1)
button8.pack(side=LEFT)

button9 = Button(root,text='Pause' ,padx=12 ,pady=23 ,bg='brown',command=PauseVideo, width=6, height=1)
button9.pack(side=LEFT)

t1 = Text(root, height = 1, width = 100 ,padx=10 ,pady=15)  # it has been used
t1.insert(INSERT,"Path of Video file .." )
t1.place(x= 550, y = 120)


t2 = Text(root, height = 1, width = 40 ,padx=10 ,pady=15)
t2.insert(INSERT,"Belonging Classes (Proposed).." )
t2.insert(tk.END , text)
t2.place(x= 790, y = 180)  # padding in pack makes spacing difference


l2 = Label(root ,text=' Want to Know About Classes Here are the List : ' ,font="Georgia 16 italic" ,bg="black" ,fg='blue', height = 1, width = 50,padx=10 ,pady=10 )
l2.place(x= 600, y = 250)

"""#### Ma'am Advise to Aparna
t4 = Text(root, height = 1, width = 15 ,padx=10 ,pady=15,bg='black',fg='blue')  # it has been used
t4.insert(INSERT,"Accuracy =" )
t4.insert(tk.END , text)
t4.place(x= 850, y = 320)"""

# t3 = Text(root, height = 1, width = 15 ,padx=10 ,pady=15,bg='black',fg='blue')  # it has been used
# t3.insert(INSERT,"Accuracy is .." )
# t3.place(x= 850, y = 320)

# if this label4 will run ..it will show all availables classes in related dataset
# label4 = Label(root,text='Available Classes :',padx=5,pady=5)
# label4.place(x= 1360, y = 120)

classes_1 = ['shoot_bow', 'shoot_gun', 'walk', 'stand', 'wave', 'sit', 'swing_baseball', 'throw', 'shoot_ball', 'pour', 'climb_stairs', 'jump', 'pick', 'run', 'push', 'pullup', 'golf', 'kick_ball', 'clap', 'brush_hair', 'catch']
#classes = ['Kicking-Side', 'Swing-SideAngle', 'SkateBoarding-Front', 'Walk-Front', 'Golf-Swing-Side', 'Swing-Bench', 'Run-Side', 'Kicking-Front', 'Lifting', 'Riding-Horse', 'Golf-Swing-Front', 'Golf-Swing-Back', 'Diving side']
dropdown_1 = OptionMenu(root,StringVar(root, "JHMDB\nClasses"),*classes_list_1) 
dropdown_1.place(x= 610, y = 320)
dropdown_2 = OptionMenu(root,StringVar(root, "KARD\nClasses"),*classes_list_2) 
dropdown_2.place(x= 750, y = 320)   
dropdown_3 = OptionMenu(root,StringVar(root, "UCF_SPORTS\nCLasses"),*classes_list_3) 
dropdown_3.place(x= 880, y = 320)   
dropdown_4 = OptionMenu(root,StringVar(root, "UT-Interaction\nClasses"),*classes_list_4)  
dropdown_4.place(x= 1050, y = 320)  
dropdown_5 = OptionMenu(root,StringVar(root, "UCF-50\nClasses"),*classes_list_5)    
dropdown_5.place(x= 1220, y = 320)


Button(text="Close",command=root.destroy).pack(side=BOTTOM)



root.mainloop()