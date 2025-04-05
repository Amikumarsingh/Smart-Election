# Import necessary libraries
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

# Function to convert text to speech
def speak(message):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(message)

# Initialize video capture for webcam
video = cv2.VideoCapture(0)

# Load pre-trained face detection model
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ensure the data directory exists
if not os.path.exists('data/'):
    os.makedirs('data/')

# Load labels and face embeddings from pre-saved files
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Load previously voted faces, if available
try:
    with open('data/voted_faces.pkl', 'rb') as f:
        VOTED_FACES = pickle.load(f)
except FileNotFoundError:
    VOTED_FACES = []

# Initialize K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Define column names for the CSV file
COL_NAMES = ['NAME', 'VOTE', 'DATE', 'TIME']

# Create a blank background image
imgBackground = np.zeros((800, 800, 3), dtype=np.uint8)

# Function to check if a face has already voted
def is_face_voted(face_embedding, threshold=0.7):
    """Check if the face has already voted using a similarity threshold."""
    if not VOTED_FACES:
        return False
    distances = knn.kneighbors(face_embedding, n_neighbors=1, return_distance=True)
    return distances[0][0][0] < threshold

# Function to check if a voter already exists in the CSV file
def check_if_exists(value):
    try:
        with open("Votes.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == value:
                    return True
    except FileNotFoundError:
        print("Votes.csv file not found.")
    return False

# Main loop for face detection and voting
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    output = None  # Initialize output to a default value
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("Votes.csv")
        
        # Draw rectangles and display the predicted name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
        attendance = [output[0], timestamp]
        
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    
    if output is not None:
        # Check if the voter has already voted
        voter_exist = check_if_exists(output[0])
        face_voted = is_face_voted(resized_img)
        
        if voter_exist or face_voted:
            speak("You have already voted.")
            break
            
        # Save the face embedding of the voter
        VOTED_FACES.append(resized_img.flatten())
        with open('data/voted_faces.pkl', 'wb') as f:
            pickle.dump(VOTED_FACES, f)

        # Handle voting options
        if k in [ord('1'), ord('2'), ord('3'), ord('4')]:
            vote_map = {ord('1'): "BJP", ord('2'): "CONGRESS", ord('3'): "AAP", ord('4'): "NOTA"}
            selected_vote = vote_map[k]
            speak("Your vote has been recorded.")
            time.sleep(5)
            
            # Save the vote to the CSV file
            if exist:
                with open("Votes.csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    attendance = [output[0], selected_vote, date, timestamp]
                    writer.writerow(attendance)
            else:
                with open("Votes.csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COL_NAMES)
                    attendance = [output[0], selected_vote, date, timestamp]
                    writer.writerow(attendance)
            
            speak("Thank you for participating in the elections.")
            break

# Release resources
video.release()
cv2.destroyAllWindows()
