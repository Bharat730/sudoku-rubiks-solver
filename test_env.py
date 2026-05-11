import cv2
import numpy as np
import torch

print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}") # Will be False on Latitude