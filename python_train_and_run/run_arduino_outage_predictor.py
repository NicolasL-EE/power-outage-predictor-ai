""" 
This code is to predict an outage on a pre-trained pytorch model that recives voltage data from an arduino (to your computers port) ...
and then uses that data to predict wether a poweroutage is likely to occur or not 

"""

import serial
import time
import re
import torch
import numpy as np
import gc

### This will only work if "run_outage_trainer" is in the same folder as this scriot, otherwise use the full file location
from run_outage_trainer import LSTMModel, SEQ_LEN, LOOKAHEAD


### USER CONFIG SECTION

### This should be the port that the arduino uploads its serial print too, look up how to find this location
SERIAL_PORT = "/dev/tty.usbserial-110"  # Example for macOS, might be COM3 on Windows

### how fast you want to read the arduinos input, for this project it can bepretty low
BAUD_RATE = 9600

### This path should be the exact location of the file "final_trained_model.pth" from github
MODEL_PATH = "/Users/colelawryshyn/Downloads/power-outage-predictor-ai/Final Version/python_train_and_run/final_trained_model.pth"



# The LSTM was trained on data scaled by 1/2, 240 down to 120
# Depending on your location you can change the scale
def scale_voltage_for_model(voltage):
    return voltage / 2.0 

#####

def main():
    # 1) Connect to the Arduino’s serial port
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)  # let Arduino reset

    # 2) Load the trained model
    device = torch.device("cpu")
    model = LSTMModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ LSTM model loaded successfully.")

    # 3) Rolling buffer for the last 180 readings (one reading every 15s)
    rolling_buffer = []

    try:
        while True:
            # Read a line if available
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8").strip()
                # Looking for lines like: "Voltage: 52.00"
                match = re.search(r"Voltage:\s*([\d.]+)", line)
                if match:
                    raw_volt_str = match.group(1)
                    actual_volt = float(raw_volt_str)
                    print(f"Received voltage from Arduino: {actual_volt:.2f} V")

                    # 4) Scale for the model, e.g. /2 if you used 0..60 scale
                    model_input_val = scale_voltage_for_model(actual_volt)

                    # Add to rolling buffer
                    rolling_buffer.append(model_input_val)
                    if len(rolling_buffer) > SEQ_LEN:
                        rolling_buffer.pop(0)

                    # Once we have at least 180 readings
                    if len(rolling_buffer) == SEQ_LEN:
                        # 5) Convert to tensor shape (1, seq_len, 1)
                        input_np = np.array(rolling_buffer, dtype=np.float32).reshape(1, SEQ_LEN, 1)
                        input_tensor = torch.tensor(input_np, dtype=torch.float32, device=device)

                        # 6) Model inference
                        with torch.no_grad():
                            output = model(input_tensor)
                            prediction = torch.argmax(output, dim=1).item()

                        # If 1 => outage predicted
                        if prediction == 1:
                            print("⚠️ Outage predicted soon! Sending '1' to Arduino.")
                            ser.write(b'1')
                        else:
                            print("✅ No outage predicted. Sending '0'.")
                            ser.write(b'0')

                        # Cleanup
                        del input_tensor, output
                        gc.collect()

    except KeyboardInterrupt:
        print("Exiting inference loop.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
