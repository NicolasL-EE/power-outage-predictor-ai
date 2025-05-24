# power-outage-predictor-ai

This project was a inspired by a first year course where a voltage power monitor was to be created in order to ...
provide power consumption to small hospitals in africa.

The code featured in this repo is a seperate project and my first code to github

  This projects goal is to demonstrate a hypothetical use case in which some sort of voltage reader, arduino here, can be connected to a computer and run through an PyTorch ML model to predict if a power outage will occur.
  
  This PyTorch model is trained on open source data provided by a Harvard Dataverse https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CLLZZM and uploaded by @AndreaPi GitHub (Andrea Panizza) on github, https://github.com/AndreaPi/Open-industrial-datasets for open source use

  For any questions or help feel free to reach out to my email nlawryhs@uwo.ca

  Thanks for looking and hope this helps!
  
# The Files
  
  arduino_voltage_reader ->
    
    Arduino_Voltage_Reader
      1. arduino ide code that takes a input into a pin and calculates the AC equivelent (RMS) before voltage drop down 
      2. It then prints this code to the serial monitor so it can be picked up by the python code

  python_train_and_run ->
  
    run_outage_trainer
      1. The full code to train an AI on a dataset to predict future outcomes
      
    run_arduino_outage_predictor
      1. The code to take a arduino port input and predict if an outage will occur
      2. This needs editing based on which port your arduino prints to
    
    run_outage_predictor
      1. This is the base code to predict outages based on any set of data, which can be changed
      2. This is a demonstration on how the code works
    
    final_trained_model
      1. Do not modify this file!
      2. This is an actual trained model that was trained with low(ish) computing power and is just a proof of concept
      3. If you wish to train your own model find trainable data (such as data linked above) and save this model to a new Location
      4. If anyone does decide to train a new model, please request to upload here as it will probably be better than the current model

  training_data ->
    
    checkpoints
      1. Just some of the checkpoints for each trained model
