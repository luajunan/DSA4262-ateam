# How to use this Pipeline for Model Training and Predicting (In Terminal/Linux Environment)

## 1. Creating a Virtual Environment

### a. Upgrading Pip
`$ python3 -m pip install --user --upgrade pip`

- Ensure that `pip` is the latest version.

### b. Installing Virtualenv

`$ python3 -m pip install --user virtualenv`

- Install `virtualenv` in local console.

### c. Creating Virtual Environment

`$ python3 -m venv env_name`

- Here, you can change `env_name` to your preferred name. For demonstration purposes, we will use `env_name` in this case.

### d. Activate/Deactivate Virtual Environment

- To activate:

`$ source env_name/bin/activate`

- To deactivate:

`$ deactivate`

## 2. Setting up virtual environment

1. `cd` into the directory that you want to work in.

2. Clone the required repository into your console.

`$ git clone https://github.com/luajunan/DSA4262-ateam.git`
- Note that the terminal might prompt you to login to your github account.
- If entering the correct password throws an error, you are required to create a `Personal Access Token (PAT)`.
- You can visit https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token for more information on how to create one.

3. If cloning is successful, the `DSA4262-ateam` folder should be generated in your directory. You may use `$ ls` to confirm that the folder is present.

4. `$ cd DSA4262-ateam/` to enter the folder's directory.

5. Download the necessary packages listed in `requirements.txt` by running

`$ python3 -m pip install -r requirements.txt`

6. Enter the working directory to start training!

`$ cd Training\ and\ predicting\ pipeline/`

## 3. Train a model with your data (To perform prediction with our pre-trained model, you may skip this step)

`$ python model_training_lgb.py <json_input_path> <info_input_path>` or `$ python model_training_xgb.py <json_input_path> <info_input_path>`

This reads the `<json_input_path>` and `<info_input_path>` file to train a model.

**Note**: Ensure that the `.json` and `.info` files are already in this directory.

It also creates 3 directories:  
- `./processed_data` which contains the training, testing and validation sets,  
- `./training_features` which contains necessary information for prediction later, and  
- `./models` which contain the trained model file for prediction.

## 4. Predict test data with your pre-trained model (LGBM / XGB)

For demo purposes, we will use `model_prediction_lgb.py`. You may switch to `model_prediction_xgb.py` if you are interested in using XGB to predict.

`$ python model_prediction_lgb.py <json_input_path> <optional: info_input_path>`

- This reads the `<json_input_path>` file and outputs the prediction to `./outputs/outputX.csv`  

- If true labels are provided in `<optional: info_input_path>`, metrics will be printed out.  

- Our finetuned model already exists in `./models`, so you may run this without training a model yourself.

## 5. To test our pipeline, you may use our `small_testset` which will run faster than the full dataset, but poorer performance may be expected.  

`$ python model_training_lgb.py small_testset/data_train.json small_testset/info_train.info`

`$ python model_prediction_lgb.py small_testset/data_test.json`
