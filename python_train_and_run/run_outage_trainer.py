
"""
Created on Fri Mar 21 20:27:48 2025

author: Nicolas Lawryshyn
"""

"""
AI_Trainer.py – Chunk-based LSTM training (with lookahead) for outage prediction.
Predicts if an outage (Voltage=0) will occur within the next LOOKAHEAD minutes.
"""

import os
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ========== CONFIGURATIONS ==========
FILE_PATH = "/Users/User1/Downloads/power-outage-predictor-ai/training_data/trained_voltage_data.csv" #Training data location 
MODEL_SAVE_PATH = "/Users/User1/Downloads/power-outage-predictor-ai/python_train_and_run/Trained_Model2.pth" # where to store final version
CHECKPOINT_DIR = "/Users/User1/Downloads/file_location/checkpoints"  # where to store partial checkpoints

SEQ_LEN = 180               # Window size for LSTM input (minutes)
LOOKAHEAD = 10              # Predict an outage if it occurs within next 10 minutes
CHUNK_ROWS = 21000           # How many CSV rows to process at once (each row has 60 min)
BATCH_SIZE = 256              # To keep RAM usage low
EPOCHS = 1                  # Number of times we train over each chunk
LR = 0.001                  # Learning rate

# Make sure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ========== LSTM MODEL ==========
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 2)  # 2 classes: 0 = No (future) outage, 1 = Future outage

    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        _, (h_n, _) = self.lstm(x)
        # h_n[-1] shape: (batch, hidden_size)
        return self.fc(h_n[-1])


# ========== DATASET FOR LOOKAHEAD ==========
class VoltageDatasetLookahead(Dataset):
    """
    x = data[idx : idx+SEQ_LEN]
    y = 1 if any future zero-voltage in next LOOKAHEAD steps, else 0.
    """
    def __init__(self, data_array, seq_len, lookahead):
        """
        :param data_array: 1D numpy array of scaled voltages
        :param seq_len: how many steps per sequence
        :param lookahead: how far ahead we look for a future outage
        """
        self.data = data_array
        self.seq_len = seq_len
        self.lookahead = lookahead

    def __len__(self):
        return len(self.data) - self.seq_len - self.lookahead

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]  # shape (seq_len,)
        future_segment = self.data[idx + self.seq_len : idx + self.seq_len + self.lookahead]
        # Label: 1 if there's at least one "0" in the next LOOKAHEAD steps
        y = 1 if (future_segment == 0).any() else 0

        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)  # (seq_len, 1)
        y_tensor = torch.tensor(y, dtype=torch.long)  # single label
        return x_tensor, y_tensor


# ========== CHUNK-BASED TRAIN FUNCTION ==========
def train_chunk(model, data_array, device, optimizer, criterion):
    """
    Given a 1D numpy array of voltages (scaled), train the model on that chunk.
    """
    # Build dataset & loader
    dataset = VoltageDatasetLookahead(data_array, SEQ_LEN, LOOKAHEAD)
    if len(dataset) <= 0:
        return  # Not enough data to train

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  - Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Cleanup
    del dataset, loader
    torch.cuda.empty_cache()
    gc.collect()


# ========== MAIN SCRIPT (TRAINING) ==========
if __name__ == "__main__":

    # -------- DEVICE (CPU only on M3 Mac) --------
    device = torch.device("cpu")

    # -------- CHECK FILE EXISTS --------
    if not os.path.exists(FILE_PATH):
        print(f"ERROR: CSV file not found: {FILE_PATH}")
        exit(1)

    print(f"✅ File found at {FILE_PATH}. Starting chunk-based training ...")

    # -------- Initialize Model, Criterion, Optimizer --------
    model = LSTMModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # If you want to RESUME from a checkpoint, load it here:
    # checkpoint_path = os.path.join(CHECKPOINT_DIR, "brownout_model_checkpoint_xxx.pth")
    # if os.path.exists(checkpoint_path):
    #     model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    #     print(f"Resumed model from {checkpoint_path}")

    leftover_scaled = np.array([], dtype=np.float32)  # keep leftover from previous chunk for continuity

    # ---- READ CSV in CHUNKS ----
    # Each CSV row has 60 minute-values. We'll expand them to a 1D array of voltages.
    # For each chunk of N rows => up to N*60 minute-values
    # We'll attach leftover from previous iteration to keep sequences intact.
    chunk_index = 0

    # Pandas "chunksize" is # of rows read each time
    for chunk_df in pd.read_csv(FILE_PATH, chunksize=CHUNK_ROWS):
        chunk_index += 1
        print(f"\n📦 Processing chunk #{chunk_index} (size={len(chunk_df)}) ...")

        # Expand each row's 60-min columns into a single list
        # (Adjust if your CSV columns differ in naming convention)
        # 1. Define the list of minute columns
        minute_cols = [f"Min {m}" for m in range(60)]

# 2. Extract those 60 columns at once from the chunk DataFrame, divide by 2,
#    and convert to a NumPy array. This results in a shape: (num_rows, 60)
        chunk_data_array = chunk_df[minute_cols].values / 2.0


# 4. Optionally cast to float32 for memory efficiency
        # 1) Flatten, cast float32
        chunk_data_array = chunk_data_array.reshape(-1).astype(np.float32)

# 2) Combine leftover without converting to list
        combined_data = np.concatenate([leftover_scaled, chunk_data_array], axis=0)

        # ---- Scale this chunk (chunk-based scaling) ----
        scaler = MinMaxScaler()
        combined_data_scaled = scaler.fit_transform(combined_data.reshape(-1, 1)).flatten()

        # ---- Train Model on this chunk ----
        print("  - Training on scaled data:", combined_data_scaled.shape)
        train_chunk(model, combined_data_scaled, device, optimizer, criterion)

        # ---- Keep leftover for next chunk (to ensure continuity) ----
        # We need the last SEQ_LEN + LOOKAHEAD - 1 points to overlap
        overlap_size = SEQ_LEN + LOOKAHEAD - 1
        if len(combined_data_scaled) >= overlap_size:
            leftover_scaled = combined_data_scaled[-overlap_size:]
        else:
            leftover_scaled = combined_data_scaled  # If chunk is too small

        # ---- Save partial checkpoint (optional) ----
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"brownout_model_checkpoint_{chunk_index}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  ✅ Checkpoint saved: {checkpoint_path}")

        # ---- Release memory ----
        gc.collect()

    # ---- Final save of the model ----
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Finished training! Model saved to {MODEL_SAVE_PATH}\n")

