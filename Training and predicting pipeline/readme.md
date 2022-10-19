# How to use this Pipeline for Model Training and Predicting

## 1. Creating a Virtual Environment

### a. Upgrading Pip
`$ python3 -m pip install --user --upgrade pip`

Ensure that `pip` is the latest version.

### b. Installing Virtualenv

`$ python3 -m pip install --user virtualenv`

Install `virtualenv` in local console.

### c. Creating Virtual Environment

`$ python3 -m venv env_name`

Here, you can change `env_name` to your preferred name. For demonstration purposes, we will use `env_name` in this case.

### d. Activate/Deactivate Virtual Environment

To activate:
`$ source env_name/bin/activate`

To deactivate:
`$ deactivate`

## 4. Setting up virtual environment

1. `cd` into the directory that you want to work in.

2. Clone the required repository into your console.

`$ git clone https://github.com/luajunan/DSA4262-ateam.git`
- Note that the terminal might prompt you to login to your github account.
- If entering the correct password throws an error, you are required to create a `Personal Access Token (PAT)`.
- You can visit https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token for more information on how to create one.


## 3. Train a model with your data

`$ python model_training.py <json_input_path> <info_input_path>`

This reads the `<json_input_path>` and `<info_input_path>` file to train a model.  

It also creates 3 directories:  
- `./processed_data` which contains the training, testing and validation sets,  
- `./training_features` which contains necessary information for prediction later, and  
- `./models` which contain the trained model file for prediction.

## 3. Predict test data with your pre-trained model

`$ python model_prediction.py <json_input_path> <optional: info_input_path>`

This reads the `<json_input_path>` file and outputs the prediction to `./outputs/outputX.csv`  

If true labels are provided in `<optional: info_input_path>`, metrics will be printed out.  

Our finetuned model already exists in `./models`, so you may run this without training a model yourself.

## 4. To test our pipeline, you may use our `small_testset` which will run faster than the full dataset, but poorer performance is to be expected.  

`$ python model_training.py small_testset/data_train.json small_testset/info_train.info`

`$ python model_prediction.py small_testset/data_test.json`
