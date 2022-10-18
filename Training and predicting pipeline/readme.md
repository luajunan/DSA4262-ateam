# How to use this Pipeline for Model Training and Predicting

## Train a model with your data

`$ python model_training.py <json_input_path> <info_input_path>`

This reads a `.json` and a `.info` file to train a model.  

It also creates 3 directories:  
- `./processed_data` which contains the training, testing and validation sets,  
- `./training_features` which contains necessary information for prediction later, and  
- `./models` which contain the trained model file for prediction.

## Predict test data with your pre-trained model

`$ python model_prediction.py <json_input_path> <optional: info_input_path>`

This reads the `.json` file and outputs the prediction to `./outputs/outputX.csv`  

If labels are provided in the form of `data.info`, metrics will be printed out.  

Our finetuned model already exists in `./models`, so you may run this without training a model yourself.

## To test our pipeline, you may use our `small_testset` which will run faster than the full dataset, but poorer performance is to be expected.  

`$ python model_training.py small_testset/data_train.json small_testset/info_train.info`

`$ python model_prediction.py small_testset/data_test.json`