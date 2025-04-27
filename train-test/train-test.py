# Importing Libraries
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import random
from sklearn.utils import shuffle
import cv2
import mediapipe
# Dataset
# Change data path accordingly
data = pd.read_excel(r"alphabet_training_data.xlsx", header=0)
data.pop("CHARACTER")
groupValue, coordinates = data.pop("GROUPVALUE"), data.copy()
coordinates = np.reshape(coordinates.values, (coordinates.shape[0], 63, 1)) #(Col, Row, Batch)
coordinates = torch.from_numpy(coordinates).float()
groupValue = torch.from_numpy(groupValue.to_numpy()).long()

# A function to move data to CUDA if available.
def to_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

# Calculate Accuracy
def calculateAccuracy(y_true, y_pred):
  
    if y_true.dim() > 1 and y_true.size(1) > 1:
        y_true = torch.argmax(y_true, dim=1)

    y_pred = y_pred.to(y_true.device)
    predicted_classes = torch.argmax(y_pred, dim=1)
    correct_predictions = (predicted_classes == y_true).float()
    accuracy = correct_predictions.sum() / len(correct_predictions)
    return accuracy


# A Function to plot accuracy graph.
def plotAccuracyGraph(trainAccuracies, valAccuracies, epoch):
  
    plt.plot(range(1, epoch + 1), trainAccuracies, 'bo-', label='Training Accuracy')
    plt.plot(range(1, epoch + 1), valAccuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# A Function to plot loss graph.
def plotLossGraph(trainLosses, valLosses):
   
    plt.plot(trainLosses, label='Training loss')
    plt.plot(valLosses, label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
 
# Model

class LandmarkCNN(nn.Module):
    
    def __init__(self):
        super().__init__()

        # Define the convolutional layers with batch normalization, activation, and pooling.
        self.cnnLayers = nn.Sequential(
            nn.Conv1d(63, 32, 3, 1, 2),
            nn.BatchNorm1d(32),  # Batch normalization for stable training.
            nn.ReLU(),  # ReLU activation for non-linearity.

            nn.Conv1d(32, 64, 3, 1, 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, 3, 1, 2),
            nn.BatchNorm1d(128),  # Output channels (128) match the filter size of the previous layer.
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Dropout to prevent over-fitting.

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

        # Define the linear layers for classification.
        self.linearLayers = nn.Sequential(
            nn.Linear(512, 26),
            nn.BatchNorm1d(26),
            nn.ReLU(),
        )

    # Define the forward pass
    def forward(self, x):
        # Pass the input through the convolutional layers.
        x = self.cnnLayers(x)

        # Flatten the output for the linear layers.
        x = x.view(x.size(0), -1)

        # Pass the flattened output through the linear layers.
        x = self.linearLayers(x)

        return x
     
# Adding Mutations     
def mutate(model, strength=0.06):
    for param in model.parameters():
        noise = torch.randn_like(param) * strength
        param.data += noise
     
# Combining two models parameters     
def crossover(parent1, parent2):
    child = copy.deepcopy(parent1)
    for p1, p2, c in zip(parent1.parameters(), parent2.parameters(), child.parameters()):
        alpha = torch.distributions.Beta(0.9, 0.1).sample().item()  # biased toward parent1
        c.data = alpha * p1.data + (1 - alpha) * p2.data
    return child   

# Training 
def train_with_evolution(coordinates, groupValue, input_size, num_classes, num_models=10,
                         num_generations=10, mutation_rate=0.1, mutation_strength = 0.06, top_k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coordinates, groupValue = shuffle(coordinates, groupValue, random_state=42)

    # Split data into training and validation (here, 80-20 split as an example)
    train_size = int(0.8 * len(coordinates))
    x_train_tensor, y_train_tensor = coordinates[:train_size], groupValue[:train_size]
    x_val_tensor, y_val_tensor = coordinates[train_size:], groupValue[train_size:]

    # Initialize population
    population = [LandmarkCNN().to(device) for _ in range(num_models)]

    retained_models = []  # This will hold the best models from each generation

    for gen in range(num_generations):
        print(f"Generation {gen + 1}/{num_generations}")
        scores = []

        # Train each model in the population
        for model in population:
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            optimizer.zero_grad()
            outputs = model(x_train_tensor.to(device))
            loss = nn.CrossEntropyLoss()(outputs, y_train_tensor.to(device))
            loss.backward()
            optimizer.step()

            scores.append((loss.item(), model))

        # Sort models by loss (best model first) and keep top_k survivors
        scores.sort(key=lambda x: x[0])
        if top_k < num_models:
            survivors = [copy.deepcopy(m) for _, m in scores[:top_k]]
        else:
            top_k = num_models
            survivors = [copy.deepcopy(m) for _, m in scores[:top_k]]
            
        # Save the best models for the next generation
        retained_models = survivors
        #print best_model accuracy
        best_model = scores[0][1].eval()
        with torch.no_grad():
            preds = best_model(x_val_tensor.to(device))
            acc = (preds.argmax(dim=1) == y_val_tensor.to(device)).float().mean()
            print(f"Accuracy: {acc.item():.4f}\n")

        # Generate a new population through crossover and mutation
        new_population = []
        for _ in range(num_models):
            p1, p2 = random.sample(survivors, 2)
            child = crossover(p1, p2)  # Implement crossover function
            if random.random() < mutation_rate:
                mutate(child, mutation_strength)  # Implement mutate function
            new_population.append(child.to(device))

        population = new_population
        
    # Evaluate the best model of the final generation
    best_model = scores[0][1].eval()
    with torch.no_grad():
        preds = best_model(x_val_tensor.to(device))
        acc = (preds.argmax(dim=1) == y_val_tensor.to(device)).float().mean()
        print(f"Final Accuracy: {acc.item():.4f}")
    return best_model    
    #return acc   

# Parameter Tuning 
def random_search_tuning(coordinates, groupValue, input_size, num_classes, num_trials=10):
    best_accuracy = 0
    best_config = None

    for trial in range(num_trials):
        # Randomly sample hyperparameters
        num_models = random.choice([10, 15, 20, 30, 40])
        num_generations = random.choice([30, 40, 50, 60, 70, 80])
        mutation_rate = random.choice([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
        mutation_strength = random.choice([0.02, 0.04, 0.05, 0.06, 0.08])
        top_k = random.choice([3, 5, 7, 10, 15, 20, 25, 30])

        print(f"\nTrial {trial+1}:")
        print(f"Models: {num_models}, Generations: {num_generations}, Mutation rate: {mutation_rate}, Strength: {mutation_strength}, Top K: {top_k}")

        # Train using these parameters
        accuracy = train_with_evolution(
            coordinates, groupValue,
            input_size=input_size,
            num_classes=num_classes,
            num_models=num_models,
            num_generations=num_generations,
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
            top_k=top_k
        )

        # Save best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = {
                'num_models': num_models,
                'num_generations': num_generations,
                'mutation_rate': mutation_rate,
                'mutation_strength': mutation_strength,
                'top_k': top_k
            }
            
    
    print("\nBest Configuration Found:")
    print(best_config)
    print(f"Best Accuracy: {best_accuracy:.4f}")
random_search_tuning(coordinates, groupValue, input_size = 63, num_classes = 26, num_trials=50)

best_model = train_with_evolution(coordinates, groupValue, input_size = 63, num_classes = 26, num_models=30,
                         num_generations=300, mutation_rate=0.3,mutation_strength = 0.08, top_k = 7)
# save model
if best_model is not None:
    torch.save(best_model.state_dict(), "evolution_model_v2.pth")
else:
    print("Model not saved. best_model is None.")
# Testing 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import seaborn as sns

# Function to plot the confusion matrix.
def plotConfusionMatrix(confusionMatrix, classNames):
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusionMatrix, annot=True, fmt='g', xticklabels=classNames, yticklabels=classNames)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


# Load the trained model.
model = LandmarkCNN()
# model.load_state_dict(torch.load("CNN_model_alphabet_SIBI.pth"))
model.load_state_dict(torch.load("evolution_model_v2.pth"))
model.eval()

# Load the testing dataset.
# data = pd.read_excel("../Data/alphabet_testing_data.xlsx", header=0)
data = pd.read_excel(r"D:\doc\collegeproject\Sign-Language-To-Text-Conversion\dataSet\asl_2\alphabet_asl_splits\alphabet_testing_data.xlsx", header=0)

data.pop("CHARACTER")  # Remove unnecessary column.
groupValue, coordinates = data.pop("GROUPVALUE"), data.copy()

# Reshape features to match model input.
coordinates = np.reshape(coordinates.values, (coordinates.shape[0], 63, 1))
coordinates = torch.from_numpy(coordinates).float()
coordinates = [coordinates]
groupValue = groupValue.to_numpy()

# Make predictions.
predictions = []
with torch.no_grad():
    for inputs in coordinates:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())

# Calculate metrics.
accuracy = accuracy_score(predictions, groupValue)
precision = precision_score(groupValue, predictions, average='weighted', zero_division=0)
recall = recall_score(groupValue, predictions, average='weighted', zero_division=0)
f1 = f1_score(groupValue, predictions, average='weighted', zero_division=0)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}")
print(f"Recall: {recall * 100:.2f}")
print(f"F1-Score: {f1 * 100:.2f}")

# Define class names for the confusion matrix.
classNames = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


# Compute the confusion matrix.
confusionMatrix = confusion_matrix(groupValue, predictions)

# Plot the confusion matrix.
plotConfusionMatrix(confusionMatrix, classNames)

# Live Gesture Recognition
# Load the model.
model = LandmarkCNN()
model.load_state_dict(torch.load("evolution_model_v2.pth"))
# model.load_state_dict(torch.load("CNN_model_number_SIBI.pth"))

# Set up video capture and MediaPipe Hands.
cap = cv2.VideoCapture(0)
handTracker = mediapipe.solutions.hands
drawing = mediapipe.solutions.drawing_utils
drawingStyles = mediapipe.solutions.drawing_styles

# Initialize the MediaPipe Hands detector.
handDetector = handTracker.Hands(static_image_mode=True, min_detection_confidence=0.2)

# Define the classes.

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

model.eval()

while True:
    ret, frame = cap.read()
    # Flip the frame horizontally for a mirrored view.
    height, width, _ = frame.shape
    # Convert the frame to RGB for MediaPipe processing.
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgMediapipe = handDetector.process(frameRGB)

    coordinates = []
    x_Coordinates = []
    y_Coordinates = []
    z_Coordinates = []

    if imgMediapipe.multi_hand_landmarks:
        for handLandmarks in imgMediapipe.multi_hand_landmarks:
            # Draw the hand landmarks on the frame.
            drawing.draw_landmarks(
                frame,
                handLandmarks,
                handTracker.HAND_CONNECTIONS,
                drawingStyles.get_default_hand_landmarks_style(),
                drawingStyles.get_default_hand_connections_style())

            data = {}
            # Extract and normalize landmark coordinates.
            for i in range(len(handLandmarks.landmark)):
                lm = handLandmarks.landmark[i]
                x_Coordinates.append(lm.x)
                y_Coordinates.append(lm.y)
                z_Coordinates.append(lm.z)

            # Apply Min-Max normalization.
            for i, landmark in enumerate(handTracker.HandLandmark):
                lm = handLandmarks.landmark[i]
                data[f'{landmark.name}_x'] = lm.x - min(x_Coordinates)
                data[f'{landmark.name}_y'] = lm.y - min(y_Coordinates)
                data[f'{landmark.name}_z'] = lm.z - min(z_Coordinates)
            coordinates.append(data)
        # Bounding box around the hand.
        x1 = int(min(x_Coordinates) * width) - 10
        y1 = int(min(y_Coordinates) * height) - 10
        x2 = int(max(x_Coordinates) * width) - 10
        y2 = int(max(y_Coordinates) * height) - 10

        predictions = []
        # Convert landmarks to model input.
        coordinates = pd.DataFrame(coordinates)
        coordinates = np.reshape(coordinates.values, (coordinates.shape[0], 63, 1))
        coordinates = torch.from_numpy(coordinates).float()

        # Predict the class.
        with torch.no_grad():
            outputs = model(coordinates)
            _, predicted = torch.max(outputs.data, 1)
            predictions = predicted.cpu().numpy()

        predicted_character = [key for key, value in classes.items() if value == predictions[0]][0]
        # Display the prediction.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
    # Show the frame.
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop if 'q' is pressed.
# Release resources.
cap.release()
cv2.destroyAllWindows()