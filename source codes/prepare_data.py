import os
import sys
import pandas as pd
import numpy as np
import json

import mmh3
import screed
import pyhash

def getKmers(sequence, size):
    return [sequence[x:x+size] for x in range(len(sequence) - size + 1)]
    
def murmur_hash(kmer):
    # calculate the reverse complement
    rc_kmer = screed.rc(kmer)
    #print(rc_kmer)
    # determine whether original k-mer or reverse complement is lesser
    if kmer < rc_kmer:
        canonical_kmer = kmer
    else:
        canonical_kmer = rc_kmer

    # calculate murmurhash using a hash seed of 42
    hash = mmh3.hash(canonical_kmer, 42)
    if hash < 0: hash += 2**32

    # done
    return hash
    
def average_hash(row, size, hashing=32, hash="fnv"):
  kmers = getKmers(row, size)
  #FNV Hash
  if hash == "fnv":
    if hashing == 32:
      hasher = pyhash.fnv1_32()
    if hashing == 64:
      hasher = pyhash.fnv1_64()

    avg = 0
    for kmer in kmers:
      avg += hasher(kmer)
    return avg/(len(row))
  

  #Murmur hash
  if hash == "murmurhash":

    avg = 0
    for kmer in kmers:
      avg += murmur_hash(kmer)
      #print(avg)
    return avg/(len(row))
    
def apply_hashing(data, all = True, size=1, hashing=32, hash="fnv"):
    #if all == True, include all the possible kmers
    i = 7
    colNames = []
    res = []

    if all:
        for j in range(1, i+1):
            colName = "avg_hash_" + str(j) + "_mer"
            colNames.append(colName)
            res.append(data.apply(lambda x: average_hash(x['nucleo_seq'], size=j, hashing = hashing, hash=hash), axis=1).tolist())
        df = pd.DataFrame(res).transpose()
        df.columns = colNames
        return df
        
    #else, only the specified k-mer is calculated. Default k-mer is 1
    else:
        colName = "avg_hash_" + str(size) + "_mer"
        colNames.append(colName)    
        res.append(data.apply(lambda x: average_hash(x['nucleo_seq'], size=size, hashing = hashing, hash=hash), axis=1).tolist())
        df = pd.DataFrame(res).transpose()
        df.columns = colNames
        return df

def train_test_split(df, labels, train_split):
  # to find out split index for 80-20 split
  init_split = int(train_split * len(df))
  print(f"initial 80% split index: {init_split}")
  last_gene = df.iloc[init_split]["gene_id"]
  print(f"last gene in train set: {last_gene}\n")

  next_gene = last_gene
  split = init_split
  while next_gene==last_gene: # break when next gene different from last gene in train set
    split += 1
    next_gene = df.iloc[split]["gene_id"]

  print(f"final split index (no overlap genes in train/test): {split}")
  print(f"first gene in test set: {next_gene}")
  return split
  
base = sorted(set(["A", "T", "C", "G"]))

def get_vec(len_doc,word):
    global base
    empty_vector = [0] * len_doc
    vect = 0
    find = np.where( np.array(base) == word)[0][0]
    empty_vector[find] = 1
    return empty_vector

def get_matrix(input):
    global base
    mat = []
    len_doc = len(base)
    for i in input:
        vec = get_vec(len_doc,i)
        mat.append(vec)
        
    return np.asarray(mat)

def oneHot(data):
    res = []
    for idx, row in data.iterrows():
        temp = [row["nucleo_seq"]] + list(np.reshape(get_matrix(list(row["nucleo_seq"])),28))
        res.append(temp)
    return res

def convert_prob_mat(df):
    res=[]
    i=0
    while i < len(list(df)):
        res.append(list(df)[i:i+4])
        i=i+4
    mat = pd.DataFrame(np.array(res).transpose(), columns=['1','2','3','4','5','6','7'])
    mat = mat.rename({0:"A",1:"C",2:"G",3:"T"}, axis="index")
    return mat
  
def log_odds(x):
    if x == 0:
        return 0
    else:
        return int(10*np.log10(x/0.25))
        
def get_PWM(seq, log_odds_dict):
    res = 0
    for i in range(len(seq)):
        base = seq[i]
        dic = log_odds_dict[str(i+1)]
        res = res + dic[base]
    return res
    
def val_split(test, test_labels, val_split):
    init_split = int(val_split * len(test))
    print(f"initial {val_split*100}% split index: {init_split}")
    last_gene = test.iloc[init_split]["gene_id"]
    print(f"last gene in train set: {last_gene}\n")

    next_gene = last_gene
    split = init_split
    while next_gene==last_gene: # break when next gene different from last gene in train set
        split += 1
        next_gene = test.iloc[split]["gene_id"]

    print(f"final split index (no overlap genes in train/validation): {split}")
    print(f"first gene in validation set: {next_gene}")
    return test[:split], test[split:], test_labels[:split], test_labels[split:]

def prepare_data(json_input_path, info_input_path):
    '''
    Inputs:
        json_input_path: Path to `data.json`
        info_input_path: Path to `data.info`
        
    Outputs:
        trainset.csv, testset.csv, valset.csv : saved in `processed_data` directory
        
    This function takes in two raw data and label files (`data.json` and `data.info`)
    and performs feature engineering steps, and outputs 3 csv files, which are used
    for training, testing and validating a machine learning model.
    '''
    # loads data
    print(f"Loading {json_input_path}...")
    data = [json.loads(line) for line in open(json_input_path)]
    print(f"Loading {info_input_path}...")
    info = pd.read_csv(info_input_path)
    
    # transfer information from json dict to list
    print("Transferring data from json to dataframe...")
    res = []
    for row in data:
        for trans_id in row.keys():
            for trans_pos in row[trans_id].keys():
                for nucleo_seq in row[trans_id][trans_pos].keys():
                    temp = list(np.mean(np.array(row[trans_id][trans_pos][nucleo_seq]), axis=0))
                    res.append([trans_id, trans_pos, nucleo_seq] + temp)
    
    # put list into dataframe with colnames
    data = pd.DataFrame(res, columns = ['transcript_id', 'transcript_pos', 'nucleo_seq',
                                      'dwell_time_-1', 'sd_-1', 'mean_-1',
                                      'dwell_time_0', 'sd_0', 'mean_0',
                                      'dwell_time_1', 'sd_1', 'mean_1'
                                     ])
    
    # convert values to int
    data['transcript_pos'] = data.transcript_pos.astype(int)
    
    # merging data.json and data.info dataframes
    print("Joining dataframes...")
    new_df = pd.merge(data, info, left_on=['transcript_id','transcript_pos'],right_on=['transcript_id','transcript_position'], how = 'left')
    
    # sort by gene_id, preparing to split
    print("Sorting by gene_id...")
    new_df = new_df.sort_values(by=['gene_id'], ascending=False).reset_index(drop=True)
    
    labels = new_df.label
    
    # feature engineering
    print("Feature engineering steps:")
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
    
    # getting the right train split
    print("Doing train test split (80-20)...")
    train_split = train_test_split(new_df, labels, 0.8)
    
    # PWM
    print("PWM...")
    colSum = oneHot_encode.iloc[:train_split].sum(axis=0)
    mat = convert_prob_mat(colSum)
    base_prob_mat = mat.div(train_split)
    
    log_odds_pos = base_prob_mat.applymap(log_odds)
    log_odds_dict = log_odds_pos.to_dict()
    PWM_col = pd.DataFrame(new_df.apply(lambda x: get_PWM(x["nucleo_seq"], log_odds_dict), axis=1), columns=["PWM"])
    
    # add feature engineering columns to dataframe
    print("Done. Adding new columns to training and testing sets...")
    X_train = pd.concat([new_df, mm_hash_kmer, PWM_col, oneHot_encode, GGACT], axis=1).iloc[:train_split]
    X_test = pd.concat([new_df, mm_hash_kmer, PWM_col, oneHot_encode, GGACT], axis=1).iloc[train_split:]
    y_train = labels[:train_split]
    y_test = labels[train_split:]
    
    # getting validation set
    print("Getting validation set(50-50)...")
    X_test, X_val, y_test, y_val = val_split(X_test, y_test, 0.5)
    
    X_test.transcript_pos = X_test.transcript_pos.astype(int)
    X_train.transcript_pos = X_train.transcript_pos.astype(int)
    X_val.transcript_pos = X_val.transcript_pos.astype(int)
    
    # Dropping unnecessary columns
    X_train.drop(columns=["transcript_id", "transcript_pos", "label", "nucleo_seq", "gene_id", "transcript_position"], inplace=True)
    X_test.drop(columns=["transcript_id", "transcript_pos", "label", "nucleo_seq", "gene_id", "transcript_position"], inplace=True)
    X_val.drop(columns=["transcript_id", "transcript_pos", "label", "nucleo_seq", "gene_id", "transcript_position"], inplace=True)
    
    X_train["label"] = y_train
    X_test["label"] = y_test
    X_val["label"] = y_val
    
    # saving to csv
    print("Saving csv files...")
    outdir = f'./processed_data`'
    if not os.path.exists(outdir):
        os.mkdir(outdir) 
    
    X_train.to_csv(path_or_buf=os.path.join(outdir, "trainset.csv"), index=False)
    X_test.to_csv(path_or_buf=os.path.join(outdir, "testset.csv"), index=False)
    X_val.to_csv(path_or_buf=os.path.join(outdir, "valset.csv"), index=False)
    
    print("Files saved successfully.")
    
if __name__ == "__main__":
    
    # usually will not run this, as it is not a requirement.
    # required pipelines are `model_training.py` and `model_prediction.py`
    
    prepare_data(json_input_path=sys.argv[1],
                info_input_path=sys.argv[2])
                
    ## run on terminal:
    ## $ python prepare_data.py <json_input_path> <info_input_path>
