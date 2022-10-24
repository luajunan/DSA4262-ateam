import numpy as np
import pandas as pd
import xgboost as xgb

from model_training_xgb import *

# Disable warning
pd.options.mode.chained_assignment = None  # default='warn'

def predict(json_input_path, *args):
    '''
    Input:
        json_input_path: Path to `data.json`
    
    Output:
        outputX.csv : probability scores of whether transcript id and transcript position is m6a site
    '''
    
    # load the data
    print(f"Loading data from {json_input_path}...")
    data = [json.loads(line) for line in open(json_input_path)]
    
    # create dataframe
    res = []
    for row in data:
        for trans_id in row.keys():
            for trans_pos in row[trans_id].keys():
                for nucleo_seq in row[trans_id][trans_pos].keys():
                    temp = list(np.mean(np.array(row[trans_id][trans_pos][nucleo_seq]), axis=0))
                    res.append([trans_id, trans_pos, nucleo_seq] + temp)
    
    # put list into dataframe with colnames
    new_df = pd.DataFrame(res, columns = ['transcript_id', 'transcript_pos', 'nucleo_seq',
                                      'dwell_time_-1', 'sd_-1', 'mean_-1',
                                      'dwell_time_0', 'sd_0', 'mean_0',
                                      'dwell_time_1', 'sd_1', 'mean_1'
                                     ])
                                     
    # feature engineering steps:
    print("Adding feature engineering columns...")
    print("One hot encoding...")
    colName = ["nucleo_seq"]
    for j in range(1, 8):
        for i in ["A", "C", "G", "T"]:
            colName.append(i + "_" + str(j))
    oneHot_encode = pd.DataFrame(oneHot(new_df), columns=colName).drop(columns=['nucleo_seq'], axis=1)
    
    print("GGACT present...")
    GGACT = new_df.nucleo_seq.str.match(".GGACT.").astype(int).to_frame(name="GGACT_Present")

    print("Hashing...")
    # fnv_7mer_hash = apply_hashing(new_df, hash="fnv", all=False, size=7)
    # fnv_hash_kmer = apply_hashing(new_df, hash="fnv")
    mm_hash_kmer = apply_hashing(new_df, hash="murmurhash")
    
    # PWM -- retrieve `log_odds_dict` from `./training_features`
    print("PWM...")
    directory = "training_features"
    path = os.path.join(directory, "log_odds_dict.json")
    with open(path) as f:
        log_odds_dict = json.load(f)
    PWM_col = pd.DataFrame(new_df.apply(lambda x: get_PWM(x["nucleo_seq"], log_odds_dict), axis=1), columns=["PWM"])
    
    
    # adding the columns
    X = pd.concat([new_df, mm_hash_kmer, PWM_col, oneHot_encode, GGACT], axis=1)
    
    ## can use try...except, but want to see code tracing for below:
    
    # load pre-trained model
    print("Data processed successfully. Loading pre-trained model...")
    clf_xgb = xgb.XGBClassifier()
    clf_xgb.load_model("models/trained_model_xgb.bin")
    print("Pre-trained model loaded successfully.")
    
    # load RFEfeatures
    print("Loading RFE features from training_features...")
    directory = "./training_features"
    path = os.path.join(directory, "RFEfeatures_xgb.json")
    with open(path) as f:
        predictive_features = json.load(f)
    print("RFE features successfully loaded.")
    
    # predicting the scores
    print("Predicting probability scores...")
    score = clf_xgb.predict_proba(X[predictive_features])[:, 1]
    
    output_features = ['transcript_id', 'transcript_pos']
    output = X[output_features]
    output['score'] = score
    print("Prediction done...")
    # rename column
    output.rename(columns={"transcript_pos":"transcript_position"}, inplace=True)
    
    # saving output to csv
    outdir = './prediction_outputs'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    num = len([entry for entry in os.listdir(outdir) if os.path.isfile(os.path.join(outdir, entry))]) + 1

    path = os.path.join(outdir, f"output{num}.csv")
    
    print(f"Saving output to {path}")
    output.to_csv(path_or_buf=path, index=False)
    
    
    ### if labels are given in *args
    if args:
        info = pd.read_csv(args[0])
        info = info.astype({'transcript_position': 'int64'})
        output = output.astype({'transcript_position': 'int64'})
        
        
        # add the labels to output
        new_df = pd.merge(output, info, left_on=['transcript_id','transcript_position'],right_on=['transcript_id','transcript_position'], how = 'left')
        
        # calculate the AUC-PR
        y_pred_proba = new_df["score"]
        y_test = new_df["label"]
        y_pred = [round(i, 0) for i in y_pred_proba]
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)

        print("-----Performance Report of XGB-----:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("AUC-ROC:", auc_score)
        print("PR-ROC:", ap)
    

if __name__ == "__main__":
    
    if len(sys.argv) == 2:
        predict(sys.argv[1])
    elif len(sys.argv) == 3:
        predict(sys.argv[1], sys.argv[2])
    else:
        raise Exception("Incorrect format or incorrect number of inputs.")
    print("Prediction complete.")
    ## run on terminal:
    ## $ python model_prediction.py <json_input_path>
    ## or if true labels is available,
    ## $ python model_prediction.py <json_input_path> <info_input_path>
