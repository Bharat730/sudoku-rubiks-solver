import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ── Model Definition ──────────────────────────────────────────────────────────

class DigitCNN(nn.Module):
    """Small CNN trained on MNIST to classify digits 0-9 (0 = empty cell)."""
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
_model = None
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ── Training ──────────────────────────────────────────────────────────────────

def train_model(save_path="models/digit_model.pth"):
    """Download MNIST and train the digit CNN. Run once."""
    from torchvision import datasets
    from torch.utils.data import DataLoader

    print("Downloading MNIST and training model...")
    train_data = datasets.MNIST("models/mnist", train=True,  download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ]))
    loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = DigitCNN().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out  = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/5  loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

# ── Loading ───────────────────────────────────────────────────────────────────

def load_model(path="models/digit_model.pth"):
    """Load trained model from disk."""
    global _model
    _model = DigitCNN().to(DEVICE)
    _model.load_state_dict(torch.load(path, map_location="cpu"))
    _model.eval()
    print("Digit model loaded.")

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

def is_empty_cell(cell_gray):
    """Return True if the cell has no digit (mostly white/blank)."""
    _, thresh = cv2.threshold(cell_gray, 128, 255, cv2.THRESH_BINARY_INV)
    # Crop inner 60% to avoid grid lines
    h, w = thresh.shape
    inner = thresh[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    return cv2.countNonZero(inner) < 20

def predict_cell(cell_bgr):
    """
    Given a single cell image (BGR), return predicted digit (0 = empty).
    """
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    if is_empty_cell(gray):
        return 0
    tensor = _transform(gray).unsqueeze(0).to(DEVICE)
    model  = get_model()
    with torch.no_grad():
        out  = model(tensor)
        pred = out.argmax(dim=1).item()
    return pred

def extract_board(cells):
    """
    Takes list of 81 cell images, returns 9x9 board as list of lists.
    """
    get_model()  # ensure model is loaded
    board = []
    for i in range(9):
        row = []
        for j in range(9):
            digit = predict_cell(cells[i * 9 + j])
            row.append(digit)
        board.append(row)
    return board