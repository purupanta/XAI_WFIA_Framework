# If needed, uncomment the following (and re-run):
# %pip install pandas numpy scikit-learn matplotlib torch torchvision torchaudio --quiet
# Optional: if you have a real xLSTM implementation available:
# %pip install xlstm-pytorch --quiet  # (example package name; adjust as needed)

# utils/deps.py

import os, math, json, random, pathlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix
)