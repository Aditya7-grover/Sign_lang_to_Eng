import streamlit as st
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# --- Title ---
st.title("Sign Language Recognition - Model Evaluation")

# --- Model Architecture ---
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

# --- Load Model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_dir = Path(__file__).resolve().parent
model_path = script_dir.parent / "model" / "evolution_model_v2.pth"
model = LandmarkCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Load Test Data ---
data_path = script_dir.parent / "data" / "alphabet_testing_data.xlsx"
data = pd.read_excel(data_path)
data.pop("CHARACTER")
groupValue = data.pop("GROUPVALUE")
coordinates = np.reshape(data.values, (data.shape[0], 63, 1))
coordinates = torch.from_numpy(coordinates).float()

# --- Make Predictions ---
predictions = []
with torch.no_grad():
    outputs = model(coordinates)
    _, predicted = torch.max(outputs.data, 1)
    predictions = predicted.cpu().numpy()

# --- Evaluation Metrics ---
accuracy = accuracy_score(groupValue, predictions)
precision = precision_score(groupValue, predictions, average='weighted', zero_division=0)
recall = recall_score(groupValue, predictions, average='weighted', zero_division=0)
f1 = f1_score(groupValue, predictions, average='weighted', zero_division=0)

# --- Displaying Accuracy vs Generation duruing training
st.write("Accuracy Vs Generation during training of model")
generations = list(range(1, 301))
accuracies = [
    0.0421, 0.0421, 0.0561, 0.0187, 0.0187, 0.0187, 0.0187, 0.0187, 0.0187, 0.0187,
0.0187, 0.0187, 0.0327, 0.0327, 0.0327, 0.0327, 0.0327, 0.0327, 0.0327, 0.0327,
0.0421, 0.0514, 0.0327, 0.0794, 0.1075, 0.0935, 0.1028, 0.0981, 0.1308, 0.1168,
0.1168, 0.1963, 0.1168, 0.1449, 0.1262, 0.1495, 0.3084, 0.1262, 0.2617, 0.2617,
0.3178, 0.2477, 0.4019, 0.5047, 0.5514, 0.6495, 0.7009, 0.7056, 0.8224, 0.7570,
0.8364, 0.8037, 0.8832, 0.8411, 0.8832, 0.8084, 0.9065, 0.8224, 0.8785, 0.8411,
0.8738, 0.8551, 0.8879, 0.8598, 0.8598, 0.8785, 0.8738, 0.8505, 0.8925, 0.8505,
0.8551, 0.8505, 0.8879, 0.9065, 0.8738, 0.8645, 0.8551, 0.9159, 0.8551, 0.8832,
0.9065, 0.9019, 0.9112, 0.8972, 0.8785, 0.8972, 0.9065, 0.8785, 0.9346, 0.8832,
0.8972, 0.8364, 0.8925, 0.8785, 0.9112, 0.8598, 0.9019, 0.8832, 0.8925, 0.8738,
0.8832, 0.9019, 0.9112, 0.8738, 0.8925, 0.8925, 0.9206, 0.8738, 0.9346, 0.8972,
0.9393, 0.8925, 0.8692, 0.9159, 0.8879, 0.9252, 0.8785, 0.9112, 0.8972, 0.9065,
0.8972, 0.8738, 0.9112, 0.9019, 0.9393, 0.8925, 0.9299, 0.9206, 0.9252, 0.9206,
0.9439, 0.8879, 0.9299, 0.9159, 0.9159, 0.9206, 0.9019, 0.8832, 0.9206, 0.8785,
0.9252, 0.8925, 0.9019, 0.9019, 0.9065, 0.9299, 0.9346, 0.9206, 0.9299, 0.9159,
0.8925, 0.9299, 0.9065, 0.9299, 0.8972, 0.9206, 0.8645, 0.9439, 0.9346, 0.9346,
0.9346, 0.9112, 0.9019, 0.8318, 0.9206, 0.8785, 0.9112, 0.8925, 0.9065, 0.8879,
0.9065, 0.9206, 0.9206, 0.8972, 0.9252, 0.9159, 0.9206, 0.9626, 0.9393, 0.9439,
0.9206, 0.9252, 0.9533, 0.9533, 0.9533, 0.9252, 0.9393, 0.9439, 0.9206, 0.8972,
0.9346, 0.9439, 0.9112, 0.9579, 0.8879, 0.9393, 0.9159, 0.9346, 0.9065, 0.8832,
0.9112, 0.8925, 0.8879, 0.9299, 0.8832, 0.9019, 0.9206, 0.9019, 0.9252, 0.9065,
0.9112, 0.9299, 0.9065, 0.9112, 0.8879, 0.9159, 0.9252, 0.8879, 0.9393, 0.9112,
0.8925, 0.9206, 0.9159, 0.9299, 0.9393, 0.9206, 0.9439, 0.9299, 0.9533, 0.9206,
0.9486, 0.9346, 0.9252, 0.9393, 0.9299, 0.9206, 0.9346, 0.9299, 0.9393, 0.9486,
0.9533, 0.9533, 0.9533, 0.9486, 0.9346, 0.9486, 0.9579, 0.9393, 0.9533, 0.9393,
0.9486, 0.9486, 0.9533, 0.9486, 0.9579, 0.9486, 0.9486, 0.9533, 0.9252, 0.9393,
0.9439, 0.9486, 0.9533, 0.9439, 0.9486, 0.9486, 0.9299, 0.9439, 0.9439, 0.9346,
0.9065, 0.9065, 0.9065, 0.8785, 0.9393, 0.9346, 0.9486, 0.9299, 0.9486, 0.9206,
0.9439, 0.9346, 0.9299, 0.9439, 0.9393, 0.9346, 0.9393, 0.9299, 0.9626, 0.9299,
0.9533, 0.9439, 0.9112, 0.9206, 0.9439, 0.9393, 0.9393, 0.9159, 0.9206, 0.9206
]

# Make a DataFrame
df = pd.DataFrame({
    'Generation': generations,
    'Accuracy': accuracies
})

# Plot
st.line_chart(df.set_index('Generation'))

# --- Display Metrics ---
st.subheader("Evaluation Metrics")
st.metric("Accuracy", f"{accuracy * 100:.2f}%")
st.metric("Precision", f"{precision * 100:.2f}%")
st.metric("Recall", f"{recall * 100:.2f}%")
st.metric("F1-Score", f"{f1 * 100:.2f}%")

# --- Confusion Matrix ---
st.subheader("Confusion Matrix")
classNames = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
cm = confusion_matrix(groupValue, predictions)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classNames, yticklabels=classNames)
plt.ylabel('Actual')
plt.xlabel('Predicted')
st.pyplot(fig)

# --- Additional Tip ---
st.info("Model: Evolution-based CNN. Data: ASL Alphabet Testing Dataset.")