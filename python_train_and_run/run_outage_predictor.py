
"""
Outage detection, using a pre-trained LSTM model, predicting an outage on an example data set
"""

import warnings
import torch
import numpy as np
import gc
import os
import sys

### This will only work if "run_outage_trainer" is in the same folder as this scriot, otherwise use the full file location
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_outage_trainer import LSTMModel, SEQ_LEN, LOOKAHEAD

                # -------- Suppress Pytorch's FutureWarning --------
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

                # -------- LOAD TRAINED MODEL --------
    ### This path should be the exact location of the file "final_trained_model.pth" from github
    MODEL_PATH = "/Users/colelawryshyn/Downloads/power-outage-predictor-ai/Final Version/python_train_and_run/final_trained_model.pth"

    device = torch.device("cpu")
    model = LSTMModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.eval()
print("✅ Model loaded for inference (brownout scenario, no zeros).")


# -------- CREATE TEST SEQUENCE --------
# This data can be changed to whatever you want, including "testing_data_2016_Jan-June" which this model has not been trained on
test_voltage_data = np.concatenate([
    [120.0,108.89, 97.78, 86.67, 75.56, 64.44,53.33, 42.22, 38.11, 10.0]
], dtype=np.float32)

# Naive scale from 0..120 => 0..1
test_scaled = test_voltage_data / 120.0


input_tensor = torch.tensor(
    test_scaled.reshape(1, -1, 1),  # Use actual data length
    dtype=torch.float32,
    device=device
)

# -------- INFERENCE --------
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

if predicted_class == 1:
    print("⚠️ Model predicts an outage/brownout soon!")
else:
    print("✅ Model predicts NO outage/brownout in the next 10 minutes.")

# Cleanup
del model, input_tensor, output
gc.collect()
print("🧹 Finished inference, memory cleared.")
