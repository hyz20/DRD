import pandas as pd
import numpy as np
import sys
import argparse
import time
from sklearn.preprocessing import Normalizer 
from sklearn.preprocessing import MinMaxScaler
sys.path.append("..")
from utils.trans_format import format_trans


def stat_prob(fin):
    t_1 = time.time()
    train_log = pd.read_json(fin + 'click_log/Train_log.json')
    train_dat = pd.read_json(fin + 'json_file/Train.json')
    train_dat = format_trans(train_dat)
    click_prob = train_log[['did','isClick']].groupby('did').mean()
    click_prob.rename(columns={'isClick':'clickProb'},inplace = True)
    train_log_fe = pd.merge(train_dat,click_prob,how='left',on=['did'])
    if 'WEB' in fin:
        print('Normalizing feature...')
        test_dat = pd.read_json(fin + 'json_file/Test.json')
        vali_dat = pd.read_json(fin + 'json_file/Vali.json')
        # test_dat = format_trans(test_dat)
        train_log_fe, test_dat, vali_dat = norm_feature(train_log_fe, test_dat, vali_dat)
        test_dat.to_json(fin + 'json_file/Test.json')
        vali_dat.to_json(fin + 'json_file/Vali.json')
    train_log_fe.to_json(fin + 'click_log/Train_log_trans.json')
    
    t_2 = time.time()
    print('Trans time: {}'.format(t_2 - t_1))
    return 0

def norm_feature(df, df_test, df_vali):
    arr = np.array(df['feature'].tolist())
    arr_test = np.array(df_test['feature'].tolist())
    arr_vali = np.array(df_vali['feature'].tolist())

    scaler = MinMaxScaler()
    scaler.fit(arr)
    # arr_norm = Normalizer(norm='l2').fit_transform(arr)
    arr_norm = scaler.transform(arr)
    arr_test_norm = scaler.transform(arr_test)
    arr_vali_norm = scaler.transform(arr_vali)

    df['norm_feature'] = arr_norm.tolist()
    df.drop('feature',axis=1, inplace=True)
    df.rename(columns={'norm_feature':'feature'}, inplace = True)

    df_test['norm_feature'] = arr_test_norm.tolist()
    df_test.drop('feature',axis=1, inplace=True)
    df_test.rename(columns={'norm_feature':'feature'}, inplace = True)

    df_vali['norm_feature'] = arr_vali_norm.tolist()
    df_vali.drop('feature',axis=1, inplace=True)
    df_vali.rename(columns={'norm_feature':'feature'}, inplace = True)

    return df, df_test, df_vali

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fin', required=True)
    #parser.add_argument('--fout', required=True)

    args = parser.parse_args()
    stat_prob(args.fin)
