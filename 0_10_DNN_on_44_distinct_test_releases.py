#!/usr/bin/env python
# encoding:utf-8
"""
Project: Object-oriented-Metric-Thresholds
File: 1_4_DNN_on_44_distinct_test_releases.py
Date : 2023/5/3 17:24
Author : njumy
HomePage : https://github.com/meiyuanqing
Email : dg1533019@smail.nju.edu.cn

在44个与验证集不一样的数据上，预测性能
应用神经网络类模块——多层感知分类器，实现分类，参考：
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

"""

# er指标中将SLOC换成loc
def predictive_performance(df_test):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix

    df_test['bugBinary'] = df_test.bug.apply(lambda x: 1 if x > 0 else 0)
    df_test['predictBinary'] = df_test.predict.apply(lambda x: 1 if x > 0 else 0)

    # confusion_matrix()函数中需要给出label, 0和1，否则该函数算不出TP,因为不知道哪个标签是poistive.
    c_matrix = confusion_matrix(df_test["bugBinary"], df_test['predictBinary'], labels=[0, 1])
    tn, fp, fn, tp = c_matrix.ravel()

    if (tn + fp) == 0:
        tnr_value = 0
    else:
        tnr_value = tn / (tn + fp)

    if (fp + tn) == 0:
        fpr = 0
    else:
        fpr = fp / (fp + tn)

    s_p, s, f_p, f = 0, 0, 0, 0

    if f != 0:
        effort_random = f_p / f
    else:
        effort_random = 0

    if s != 0:
        effort_m = s_p / s
    else:
        effort_m = 0

    if effort_random != 0:
        er = (effort_random - effort_m) / effort_random
    else:
        er = 0

    try:
        auc_value = roc_auc_score(df_test['bugBinary'], df_test['predictBinary'], labels=[0, 1])
    except Exception as err1:
        print(err1)
        auc_value = 0.5
    recall_value = recall_score(df_test['bugBinary'], df_test['predictBinary'], labels=[0, 1])
    precision_value = precision_score(df_test['bugBinary'], df_test['predictBinary'], labels=[0, 1])
    f1_value = f1_score(df_test['bugBinary'], df_test['predictBinary'], labels=[0, 1])
    gm_value = (recall_value * tnr_value) ** 0.5
    pdr = recall_value
    pfr = fpr  # fp / (fp + tn)
    bpp_value = 1 - (((0 - pfr) ** 2 + (1 - pdr) ** 2) * 0.5) ** 0.5

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    error_rate = (fp + fn) / (tp + fp + fn + tn)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    valueOfbugBinary = df_test["predictBinary"].value_counts()  # 0 和 1 的各自的个数

    if len(valueOfbugBinary) <= 1:
        if valueOfbugBinary.keys()[0] == 0:
            value_0 = valueOfbugBinary[0]
            value_1 = 0
        else:
            value_0 = 0
            value_1 = valueOfbugBinary[1]
    else:
        value_0 = valueOfbugBinary[0]
        value_1 = valueOfbugBinary[1]

    if auc_value > 1 or auc_value < 0:
        auc_value = 0.5
    elif auc_value < 0.5:
        auc_value = 1 - auc_value
    Q1 = auc_value / (2 - auc_value)
    Q2 = 2 * auc_value * auc_value / (1 + auc_value)
    auc_value_variance = auc_value * (1 - auc_value) + (value_1 - 1) * (Q1 - auc_value * auc_value) \
                         + (value_0 - 1) * (Q2 - auc_value * auc_value)
    auc_value_variance = auc_value_variance / (value_0 * value_1)

    return precision_value, recall_value, auc_value, auc_value_variance, gm_value, f1_value, bpp_value, fpr, \
           tnr_value, accuracy, error_rate, mcc


def dnn_cvdp(working_dir, result_dir):
    # import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn import tree
    from sklearn.neural_network import MLPClassifier

    # display all columns and rows, and set the item of row of dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    # display all columns and rows, and set the item of row of dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    # display all rows and columns of a matrix
    np.set_printoptions(threshold=np.sys.maxsize, linewidth=np.sys.maxsize)

    dir_validation = working_dir + 'data_defects_java_validtaing/'
    dir_testing = working_dir + 'data_defects_java_testing_distinct/'
    # dir_testing = working_dir + 'data_defects_java_testing/'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # 22 object-oriented features and a dependent variable:  'bug'
    metrics_22 = ['LCOM1', 'LCOM2', 'LCOM3', 'ICH', 'NHD', 'OCAIC', 'OCMIC', 'OMMIC', 'CBO', 'DAC', 'ICP', 'MPC',
                  'NIHICP', 'RFC', "NMA", "NA", "NAIMP", "NM", "NMIMP", "NumPara", "SLOC", "stms"]

    metrics_26 = ['LCOM1', 'LCOM2', 'LCOM3', 'LCOM5', 'NewLCOM5', 'ICH', 'CAMC', 'NHD', 'SNHD', 'OCAIC', 'OCMIC',
                  'OMMIC', 'CBO', 'DAC', 'ICP', 'MPC', 'NIHICP', 'RFC', 'NMA', "NA", "NAIMP", "NM", "NMIMP",
                  "NumPara", "SLOC", "stms"]

    cohesion = ['LCOM1', 'LCOM2', 'LCOM3', 'LCOM4', 'Co', 'NewCo', 'LCOM5', 'NewLCOM5', 'TCC', 'LCC', 'ICH', 'OCC',
                'PCC', 'DCd', 'DCi', 'CAMC', 'NHD', 'SNHD']

    coupling = ['ACAIC', 'ACMIC', 'AMMIC', 'DMMEC', 'OCAEC', 'OCAIC', 'OCMEC', 'OCMIC', 'OMMEC', 'OMMIC', 'DCAEC',
                'DCMEC', 'CBI', 'CBO', 'DAC', 'ICP', 'IHICP', 'MPC', 'NIHICP', 'RFC']

    inheritance = ["AID", "CLD", "DIT", "DP", "DPA", "DPD", "NMA", "NMI", "NMO", "NOA", "NOC", "NOD", "NOP", "SIX",
                   "SP", "SPA", "SPD"]

    size = ["NA", "NAIMP", "NM", "NMIMP", "NumPara", "SLOC", "stms"]

    metrics = cohesion + coupling + inheritance + size

    # store predictive performance of all versions
    df_dnn = pd.DataFrame(
        columns=['project', 'current_version', 'Sample_size', 'precision', 'recall', 'auc', 'auc_var', 'gm', 'f1',
                 'bpp', 'fpr', 'tnr', 'accuracy', 'error_rate', 'mcc'], dtype=object)

    for root, dirs, files in os.walk(dir_validation):

        for name in files:

            if name.split('-')[0] == 'bigtop':
                continue

            # df_name for spv
            print(name)
            df_training = pd.read_csv(dir_validation + name)
            file_testing = ''
            for root_t, dirs_t, files_t in os.walk(dir_testing):
                for file_t in files_t:
                    if file_t.split('-')[0][9:] == name.split('-')[0]:
                        file_testing = file_t

            if file_testing == '':
                continue
            df_testing = pd.read_csv(dir_testing + file_testing)

            print(name, file_testing)

            for metric in metrics:
                df_training = df_training[~df_training[metric].isin(['undef', 'undefined'])].reset_index(drop=True)
                df_training = df_training[~df_training[metric].isnull()].reset_index(drop=True)
                df_training = df_training[~df_training['SLOC'].isin([0])].reset_index(drop=True)

                df_testing = df_testing[~df_testing[metric].isin(['undef', 'undefined'])].reset_index(drop=True)
                df_testing = df_testing[~df_testing[metric].isnull()].reset_index(drop=True)
                df_testing = df_testing[~df_testing['SLOC'].isin([0])].reset_index(drop=True)

            df_training['bugBinary'] = df_training.bug.apply(lambda x: 1 if x > 0 else 0)

            clf = MLPClassifier(random_state=1, max_iter=300).fit(df_training.loc[:, metrics], df_training.loc[:, 'bugBinary'])
            df_testing['predict'] = clf.predict(df_testing.loc[:, metrics])
            precision, recall, auc, auc_var, gm, f1, bpp, fpr, tnr, accuracy, error_rate, mcc = \
                predictive_performance(df_testing)

            df_dnn = df_dnn.append(
                {'project': name.split('-')[0], 'current_version': file_testing[:-4], 'Sample_size': len(df_testing),
                 'precision': precision, 'recall': recall, 'auc': auc, 'auc_var': auc_var, 'gm': gm, 'f1': f1,
                 'bpp': bpp, 'fpr': fpr, 'tnr': tnr, 'accuracy': accuracy, 'error_rate': error_rate, 'mcc': mcc},
                ignore_index=True)

            df_dnn.to_csv(result_dir + 'dnn_cvdp_all_versions.csv', index=False)
            # break


if __name__ == '__main__':
    import os
    import sys
    import csv
    import math
    import time
    import random
    import shutil
    from datetime import datetime
    import pandas as pd
    import numpy as np

    s_time = time.time()

    work_Directory = "F:/PERCEIVER/pyDataSet/"
    result_Directory = "F:/PERCEIVER/DNN_on_62_metrics_data_44_distinct_test_releases/"

    print(os.getcwd())
    os.chdir(work_Directory)
    print(work_Directory)
    print(os.getcwd())

    dnn_cvdp(work_Directory, result_Directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")