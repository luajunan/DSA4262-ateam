# How to use this Pipeline for Model Training and Predicting

## Train a model with your data

`$ python model_training.py <json_input_path> <info_input_path>

This reads a `.json` and a `.info` file to train a model.  
It also creates two directories under `./processed_data` which contains the training, testing and validation sets,  
and `./training_features` which contains necessary information for prediction later on.


## Predict test data with your pre-trained model

`$ python model_prediction.py <json_input_path>`

This reads a `.json` file and outputs the prediction to `./outputs/outputX.csv`