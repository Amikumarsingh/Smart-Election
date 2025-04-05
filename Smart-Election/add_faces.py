import cv2
import pickle
import numpy as np
import os

# Ensure the data directory exists
if not os.path.exists('data/'):
    os.makedirs('data/')

# Initialize video capture for webcam
video = cv2.VideoCapture(0)

# Load pre-trained face detection model
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# List to store face data
faces_data = []

# Input the user's unique identifier (e.g., Aadhar number)
name = input("Enter your Aadhar number: ")

# Define constants for capturing frames
framesTotal = 51  # Total number of frames to capture
captureAfterFrame = 2  # Capture every nth frame

# Counter for frame processing
i = 0

# Main loop for capturing face data
while True:
    ret, frame = video.read()  # Read a frame from the webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = facedetect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the frame

    for (x, y, w, h) in faces:
        # Crop and resize the detected face
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))

        # Append the resized face to the list if conditions are met
        if len(faces_data) <= framesTotal and i % captureAfterFrame == 0:
            faces_data.append(resized_img)

        i += 1  # Increment frame counter

        # Display the number of captured faces on the frame
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    # Display the frame with annotations
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed or the required number of frames is captured
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) >= framesTotal:
        break

# Release the video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

# Convert the list of face data to a NumPy array and reshape it
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape((framesTotal, -1))
print(faces_data)

# Save the user's name (Aadhar number) to the names.pkl file
if 'names.pkl' not in os.listdir('data/'):
    # If the file doesn't exist, create it and save the names
    names = [name] * framesTotal
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    # If the file exists, load existing names, append the new names, and save
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * framesTotal
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# Save the face data to the faces_data.pkl file
if 'faces_data.pkl' not in os.listdir('data/'):
    # If the file doesn't exist, create it and save the face data
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    # If the file exists, load existing face data, append the new data, and save
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)