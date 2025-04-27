# Importing libraries
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import time

# Information 
with st.sidebar:
    st.header("App Information")
    # Add a selectbox for user to choose information type
    info_choice = st.selectbox("Select Information", ["Steps", "About", "How it Works", "Model Info"])
    # Display content based on the user's selection
    if info_choice == "About":
        st.subheader("About")
        st.write("""
        This app uses a machine learning model to recognize American Sign Language (ASL) gestures and convert them to English text.
        - The webcam will start when you click "Start Webcam."
        - The model processes the gestures and displays predictions.
        """)
    
    elif info_choice == "Steps":
        st.subheader("Steps")
        st.write("""
        1. Press 'Start Webcam' to begin capturing hand gestures from the webcam.
        2. Hold the hand gesture for 3 seconds for prediction.
        3. The predicted English word will appear on the screen.
        4. Use the 'Stop Webcam' button to stop the webcam.
        """)
    
    elif info_choice == "How it Works":
        st.subheader("How it Works")
        st.write("""
        The app captures hand gestures using the webcam and processes them through a machine learning model built with PyTorch.
        - The model uses a CNN architecture to recognize hand gestures and map them to English characters.
        - The prediction occurs when a gesture is held for 3 seconds, providing time for the model to confidently recognize the gesture.
        """)
    elif info_choice == "Model Info":
        st.subheader("Model Info")
        st.write("""
        The model is based on a Convolutional Neural Network (CNN) architecture designed to recognize American Sign Language (ASL) 
        hand gestures.
        - The network takes in 63 hand landmark coordinates and outputs a predicted character.
        - The model is trained using data collected from various hand gestures, and the training uses a classification output with 26 
          classes (A-Z).
		- To train the model evolution inspired approach is used giving more robust outcomes.
        """)
# Model definition
class LandmarkCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnnLayers = nn.Sequential(
            nn.Conv1d(63, 32, 3, 1, 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, 1, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, 3, 1, 2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Conv1d(128, 256, 3, 1, 2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 512, 5, 1, 2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(512, 512, 5, 1, 2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
        )
        self.linearLayers = nn.Sequential(
            nn.Linear(512, 26),
            nn.BatchNorm1d(26),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnnLayers(x)
        x = x.view(x.size(0), -1)
        x = self.linearLayers(x)
        return x

# Load model
model = LandmarkCNN()
model.load_state_dict(torch.load("evolution_model_v2.pth", map_location=torch.device('cpu')))
model.eval()

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
handDetector = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# Class dictionary
classes = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
    'L': 11,
    'M': 12,
    'N': 13,
    'O': 14,
    'P': 15,
    'Q': 16,
    'R': 17,
    'S': 18,
    'T': 19,
    'U': 20,
    'V': 21,
    'W': 22,
    'X': 23,
    'Y': 24,
    'Z': 25
}
# Store the predicted characters to form words
predicted_word = []

# Timer tracking for holding gestures
last_predicted_character = None
hold_start_time = None
hold_threshold = 3  # seconds to hold a gesture before adding it to the word

# Start webcam button
start = st.button('Start Webcam', key="start_button")
if "start_state" not in st.session_state:
    st.session_state.start_state = False
	
# Stop webcam button (only appears when webcam is active)
if start or st.session_state.start_state:
    st.session_state.start_state = True
    # Display the "Stop Webcam and Clear_word" button once the webcam starts
    stop = st.button('Stop Webcam', key="stop_button")
    clear_word = st.button('Clear Word', key="clear_word_button")  # Button to clear word
    # clear = st.button('Clear Output', key="clear_output")
    
  
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()  # Placeholder for video frames
    prediction_char_placeholder = st.empty()  # Placeholder for prediction alphabet
    prediction_word_placeholder = st.empty()  # Placeholder for prediction word
    if stop:
        cap.release()
        st.stop()  # Stop the Streamlit execution to prevent further actions
	
    # Handle Clear Word button click
    if clear_word:
        predicted_word.clear()  # Clear the predicted word
        prediction_word_placeholder.markdown(f"### Current Word: **{''.join(predicted_word)}**")
		
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgMediapipe = handDetector.process(frameRGB)

        coordinates = []
        x_Coordinates = []
        y_Coordinates = []
        z_Coordinates = []

        predicted_character = ""

        if imgMediapipe.multi_hand_landmarks:
            for handLandmarks in imgMediapipe.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    handLandmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(50,255,0), thickness=2))

                data = {}
                for i in range(len(handLandmarks.landmark)):
                    lm = handLandmarks.landmark[i]
                    x_Coordinates.append(lm.x)
                    y_Coordinates.append(lm.y)
                    z_Coordinates.append(lm.z)

                for i, landmark in enumerate(mp_hands.HandLandmark):
                    lm = handLandmarks.landmark[i]
                    data[f'{landmark.name}_x'] = lm.x - min(x_Coordinates)
                    data[f'{landmark.name}_y'] = lm.y - min(y_Coordinates)
                    data[f'{landmark.name}_z'] = lm.z - min(z_Coordinates)
                coordinates.append(data)

            coordinates = pd.DataFrame(coordinates)
            coordinates = np.reshape(coordinates.values, (coordinates.shape[0], 63, 1))
            coordinates = torch.from_numpy(coordinates).float()

            with torch.no_grad():
                outputs = model(coordinates)
                _, predicted = torch.max(outputs.data, 1)
                predictions = predicted.cpu().numpy()

            predicted_character = [key for key, value in classes.items() if value == predictions[0]][0]

            if predicted_character == last_predicted_character:
				# Detect hold time for the gesture
                if hold_start_time is None:
                    hold_start_time = time.time()  # Start the timer when the gesture is detected
            else:
                # Reset the timer if the gesture changes
                hold_start_time = None
                last_predicted_character = predicted_character

            # If the gesture is held for more than 3 seconds, add the letter to the word
            if hold_start_time is not None and time.time() - hold_start_time >= hold_threshold:
                predicted_word.append(predicted_character)
                last_predicted_character = None  # Reset after adding the character
                hold_start_time = None  # Reset timer after adding the letter
            # Draw bounding box and prediction
            x1 = int(min(x_Coordinates) * width) - 10
            y1 = int(min(y_Coordinates) * height) - 10
            x2 = int(max(x_Coordinates) * width) + 10
            y2 = int(max(y_Coordinates) * height) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
            cv2.putText(frame, predicted_character, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Show frame in Streamlit
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
        # Show prediction
        if predicted_character:
            prediction_char_placeholder.markdown(f"### Predicted Character: **{predicted_character}**")
        if predicted_word:
            prediction_word_placeholder.markdown(f"### Current Word: **{''.join(predicted_word)}**")
			
	    