import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ── Model Definition ──────────────────────────────────────────────────────────

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop  = nn.Dropout(0.25)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

# ── Globals ───────────────────────────────────────────────────────────────────

_model = None
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ── Training ──────────────────────────────────────────────────────────────────

def train_model(save_path="models/digit_model.pth"):
    from torchvision import datasets
    from torch.utils.data import DataLoader

    print("Downloading MNIST and training model...")
    train_data = datasets.MNIST("models/mnist", train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ]))
    loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = DigitCNN()
    opt     = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        total = 0
        for imgs, labels in loader:
            opt.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1}/5  loss: {total/len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

# ── Loading ───────────────────────────────────────────────────────────────────

def load_model(path="models/digit_model.pth"):
    global _model
    _model = DigitCNN()
    _model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    _model.eval()

def get_model():
    global _model
    if _model is None:
        import os
        if os.path.exists("models/digit_model.pth"):
            load_model()
        else:
            _model = train_model()
            _model.eval()
    return _model

# ── Cell Processing ───────────────────────────────────────────────────────────

def preprocess_cell(cell_bgr):
    """
    Clean a single cell image for digit recognition.
    Returns a clean grayscale 28x28 image ready for the model.
    """
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)

    # Crop 15% border to remove grid lines
    h, w = gray.shape
    margin_h, margin_w = int(h * 0.15), int(w * 0.15)
    gray = gray[margin_h:h-margin_h, margin_w:w-margin_w]

    # Threshold to get clean black digit on white background
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return thresh

def is_empty_cell(thresh):
    """Return True if cell has no digit."""
    # Check center region only
    h, w = thresh.shape
    cx, cy = w // 2, h // 2
    region = thresh[cy-8:cy+8, cx-8:cx+8]
    return cv2.countNonZero(region) < 15

def predict_cell(cell_bgr):
    """Given a single cell image (BGR), return predicted digit (0 = empty)."""
    thresh = preprocess_cell(cell_bgr)

    if is_empty_cell(thresh):
        return 0

    # Resize to 28x28 for model
    resized = cv2.resize(thresh, (28, 28))

    tensor = _transform(resized).unsqueeze(0)
    model  = get_model()
    with torch.no_grad():
        out  = model(tensor)
        pred = out.argmax(dim=1).item()

    # If model predicts 0 for a non-empty cell, treat as empty
    return pred if pred != 0 else 0

def extract_board(cells):
    """Takes list of 81 cell images, returns 9x9 board."""
    get_model()
    board = []
    for i in range(9):
        row = []
        for j in range(9):
            digit = predict_cell(cells[i * 9 + j])
            row.append(digit)
        board.append(row)
    return board