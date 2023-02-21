#!/usr/bin/env python
# encoding:utf-8
"""
Project: Object-oriented-Metric-Thresholds
File: 0_2_cla_on_nam_data.py
Date : 2023/1/23 17:41


由于NAM等人的方法为了与有监督方法比较，应用了二折500次的平均值比较，这个在讨论中进行二折500次的平均值比较。
但在RQ1中，只用PERCEVIER方法与CLA方法比较，并不需要二折500次的平均值比较。
"""




# 比predict_score_performance增加precision,recall,auc，以方便与NAM等人的结果比较
# 分别应用pred_22的投票得分，用0.1,0.2,...,0.9 九种对应百分比分位数作为阈值分别计算分类性能
def predict_score_performance_precision(df_test):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
    precision_1, recall_1, f1_1, auc_1, gm_1, bpp_1, mcc_1, precision_2, recall_2, f1_2, auc_2, gm_2, bpp_2, mcc_2, \
    precision_3, recall_3, f1_3, auc_3, gm_3, bpp_3, mcc_3, precision_4, recall_4, f1_4, auc_4, gm_4, bpp_4, mcc_4, \
    precision_5, recall_5, f1_5, auc_5, gm_5, bpp_5, mcc_5, precision_6, recall_6, f1_6, auc_6, gm_6, bpp_6, mcc_6, \
    precision_7, recall_7, f1_7, auc_7, gm_7, bpp_7, mcc_7, precision_8, recall_8, f1_8, auc_8, gm_8, bpp_8, mcc_8, \
    precision_9, recall_9, f1_9, auc_9, gm_9, bpp_9, mcc_9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                                             0, 0, 0, 0, 0, 0

    for i in range(9):

        df_predict = df_test.copy()

        for j in range(len(df_predict)):
            if df_predict.loc[j, 'pred_22'] >= df_predict.pred_22.quantile(0.1 * (i + 1)):
                df_predict.loc[j, 'predictBinary'] = 1
            else:
                df_predict.loc[j, 'predictBinary'] = 0

        c_matrix = confusion_matrix(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        tn, fp, fn, tp = c_matrix.ravel()
        # print(tn, fp, fn, tp)
        if (tn + fp) == 0:
            tnr_value = 0
        else:
            tnr_value = tn / (tn + fp)

        if (fp + tn) == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)

        auc_value = roc_auc_score(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        recall_value = recall_score(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        precision_value = precision_score(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])

        # recall_value = recall_score(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        f1_value = f1_score(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        gm_value = (recall_value * tnr_value) ** 0.5
        pdr = recall_value
        pfr = fpr  # fp / (fp + tn)
        bpp_value = 1 - (((0 - pfr) ** 2 + (1 - pdr) ** 2) * 0.5) ** 0.5
        mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

        if i == 0:
            precision_1, recall_1, f1_1, auc_1, gm_1, bpp_1, mcc_1 = \
                precision_value, recall_value, f1_value, auc_value, gm_value, bpp_value, mcc
        elif i == 1:
            precision_2, recall_2, f1_2, auc_2, gm_2, bpp_2, mcc_2 = \
                precision_value, recall_value, f1_value, auc_value, gm_value, bpp_value, mcc
        elif i == 2:
            precision_3, recall_3, f1_3, auc_3, gm_3, bpp_3, mcc_3 = \
                precision_value, recall_value, f1_value, auc_value, gm_value, bpp_value, mcc
        elif i == 3:
            precision_4, recall_4, f1_4, auc_4, gm_4, bpp_4, mcc_4 = \
                precision_value, recall_value, f1_value, auc_value, gm_value, bpp_value, mcc
        elif i == 4:
            precision_5, recall_5, f1_5, auc_5, gm_5, bpp_5, mcc_5 = \
                precision_value, recall_value, f1_value, auc_value, gm_value, bpp_value, mcc
        elif i == 5:
            precision_6, recall_6, f1_6, auc_6, gm_6, bpp_6, mcc_6 = \
                precision_value, recall_value, f1_value, auc_value, gm_value, bpp_value, mcc
        elif i == 6:
            precision_7, recall_7, f1_7, auc_7, gm_7, bpp_7, mcc_7 = \
                precision_value, recall_value, f1_value, auc_value, gm_value, bpp_value, mcc
        elif i == 7:
            precision_8, recall_8, f1_8, auc_8, gm_8, bpp_8, mcc_8 = \
                precision_value, recall_value, f1_value, auc_value, gm_value, bpp_value, mcc
        elif i == 8:
            precision_9, recall_9, f1_9, auc_9, gm_9, bpp_9, mcc_9 = \
                precision_value, recall_value, f1_value, auc_value, gm_value, bpp_value, mcc

    return precision_1, recall_1, f1_1, auc_1, gm_1, bpp_1, mcc_1, \
           precision_2, recall_2, f1_2, auc_2, gm_2, bpp_2, mcc_2, \
           precision_3, recall_3, f1_3, auc_3, gm_3, bpp_3, mcc_3, \
           precision_4, recall_4, f1_4, auc_4, gm_4, bpp_4, mcc_4, \
           precision_5, recall_5, f1_5, auc_5, gm_5, bpp_5, mcc_5, \
           precision_6, recall_6, f1_6, auc_6, gm_6, bpp_6, mcc_6, \
           precision_7, recall_7, f1_7, auc_7, gm_7, bpp_7, mcc_7, \
           precision_8, recall_8, f1_8, auc_8, gm_8, bpp_8, mcc_8, \
           precision_9, recall_9, f1_9, auc_9, gm_9, bpp_9, mcc_9


# 分别应用pred_22的投票得分，用0.1,0.2,...,0.9 九种对应百分比分位数作为阈值分别计算分类性能
def predict_score_performance(df_test):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
    f1_1, gm_1, bpp_1, mcc_1, f1_2, gm_2, bpp_2, mcc_2, f1_3, gm_3, bpp_3, mcc_3, f1_4, gm_4, bpp_4, mcc_4, f1_5, gm_5, \
    bpp_5, mcc_5, f1_6, gm_6, bpp_6, mcc_6, f1_7, gm_7, bpp_7, mcc_7, f1_8, gm_8, bpp_8, mcc_8, f1_9, gm_9, bpp_9, \
    mcc_9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for i in range(9):

        df_predict = df_test.copy()

        for j in range(len(df_predict)):
            if df_predict.loc[j, 'pred_22_score'] >= df_predict.pred_22_score.quantile(0.1 * (i + 1)):
                df_predict.loc[j, 'predictBinary'] = 1
            else:
                df_predict.loc[j, 'predictBinary'] = 0

        c_matrix = confusion_matrix(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        tn, fp, fn, tp = c_matrix.ravel()
        # print(tn, fp, fn, tp)
        if (tn + fp) == 0:
            tnr_value = 0
        else:
            tnr_value = tn / (tn + fp)

        if (fp + tn) == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)

        recall_value = recall_score(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        f1_value = f1_score(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        gm_value = (recall_value * tnr_value) ** 0.5
        pdr = recall_value
        pfr = fpr  # fp / (fp + tn)
        bpp_value = 1 - (((0 - pfr) ** 2 + (1 - pdr) ** 2) * 0.5) ** 0.5
        mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

        if i == 0:
            f1_1, gm_1, bpp_1, mcc_1 = f1_value, gm_value, bpp_value, mcc
        elif i == 1:
            f1_2, gm_2, bpp_2, mcc_2 = f1_value, gm_value, bpp_value, mcc
        elif i == 2:
            f1_3, gm_3, bpp_3, mcc_3 = f1_value, gm_value, bpp_value, mcc
        elif i == 3:
            f1_4, gm_4, bpp_4, mcc_4 = f1_value, gm_value, bpp_value, mcc
        elif i == 4:
            f1_5, gm_5, bpp_5, mcc_5 = f1_value, gm_value, bpp_value, mcc
        elif i == 5:
            f1_6, gm_6, bpp_6, mcc_6 = f1_value, gm_value, bpp_value, mcc
        elif i == 6:
            f1_7, gm_7, bpp_7, mcc_7 = f1_value, gm_value, bpp_value, mcc
        elif i == 7:
            f1_8, gm_8, bpp_8, mcc_8 = f1_value, gm_value, bpp_value, mcc
        elif i == 8:
            f1_9, gm_9, bpp_9, mcc_9 = f1_value, gm_value, bpp_value, mcc

    return f1_1, gm_1, bpp_1, mcc_1, f1_2, gm_2, bpp_2, mcc_2, f1_3, gm_3, bpp_3, mcc_3, f1_4, gm_4, bpp_4, mcc_4, \
           f1_5, gm_5, bpp_5, mcc_5, f1_6, gm_6, bpp_6, mcc_6, f1_7, gm_7, bpp_7, mcc_7, f1_8, gm_8, bpp_8, mcc_8, \
           f1_9, gm_9, bpp_9, mcc_9


# 分别应用pred_22_score的投票得分，用0.1,0.2,...,0.9 九种对应百分比分位数作为阈值分别计算分类性能
def do_predict(df_test):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
    f1_1, gm_1, bpp_1, f1_2, gm_2, bpp_2, f1_3, gm_3, bpp_3, f1_4, gm_4, bpp_4, f1_5, gm_5, bpp_5, \
    f1_6, gm_6, bpp_6, f1_7, gm_7, bpp_7, f1_8, gm_8, bpp_8, f1_9, gm_9, bpp_9 = \
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for i in range(9):

        df_predict = df_test.copy()

        for j in range(len(df_predict)):
            if df_predict.loc[j, 'pred_22_score'] >= df_predict.pred_22_score.quantile(0.1 * (i + 1)):
                df_predict.loc[j, 'predictBinary'] = 1
            else:
                df_predict.loc[j, 'predictBinary'] = 0

        c_matrix = confusion_matrix(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        tn, fp, fn, tp = c_matrix.ravel()
        # print(tn, fp, fn, tp)
        if (tn + fp) == 0:
            tnr_value = 0
        else:
            tnr_value = tn / (tn + fp)

        if (fp + tn) == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)

        recall_value = recall_score(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        f1_value = f1_score(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        gm_value = (recall_value * tnr_value) ** 0.5
        pdr = recall_value
        pfr = fpr  # fp / (fp + tn)
        bpp_value = 1 - (((0 - pfr) ** 2 + (1 - pdr) ** 2) * 0.5) ** 0.5

        if i == 0:
            f1_1, gm_1, bpp_1 = f1_value, gm_value, bpp_value
        elif i == 1:
            f1_2, gm_2, bpp_2 = f1_value, gm_value, bpp_value
        elif i == 2:
            f1_3, gm_3, bpp_3 = f1_value, gm_value, bpp_value
        elif i == 3:
            f1_4, gm_4, bpp_4 = f1_value, gm_value, bpp_value
        elif i == 4:
            f1_5, gm_5, bpp_5 = f1_value, gm_value, bpp_value
        elif i == 5:
            f1_6, gm_6, bpp_6 = f1_value, gm_value, bpp_value
        elif i == 6:
            f1_7, gm_7, bpp_7 = f1_value, gm_value, bpp_value
        elif i == 7:
            f1_8, gm_8, bpp_8 = f1_value, gm_value, bpp_value
        elif i == 8:
            f1_9, gm_9, bpp_9 = f1_value, gm_value, bpp_value

    return f1_1, gm_1, bpp_1, f1_2, gm_2, bpp_2, f1_3, gm_3, bpp_3, f1_4, gm_4, bpp_4, f1_5, gm_5, bpp_5, \
           f1_6, gm_6, bpp_6, f1_7, gm_7, bpp_7, f1_8, gm_8, bpp_8, f1_9, gm_9, bpp_9


def median_threshold_on_current_version(working_dir, result_dir):
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

    # dir_data = working_dir + 'data_defects_java_testing/'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # store predictive performance of all versions
    df_median = pd.DataFrame(
        columns=['project', 'current_version', 'Sample_size', 'precision_1', 'recall_1', 'f1_1', 'auc_1', 'gm_1',
                 'bpp_1', 'mcc_1', 'precision_2', 'recall_2', 'f1_2', 'auc_2', 'gm_2', 'bpp_2', 'mcc_2',
                 'precision_3', 'recall_3', 'f1_3', 'auc_3', 'gm_3', 'bpp_3', 'mcc_3',
                 'precision_4', 'recall_4', 'f1_4', 'auc_4', 'gm_4', 'bpp_4', 'mcc_4',
                 'precision_5', 'recall_5', 'f1_5', 'auc_5', 'gm_5', 'bpp_5', 'mcc_5',
                 'precision_6', 'recall_6', 'f1_6', 'auc_6', 'gm_6', 'bpp_6', 'mcc_6',
                 'precision_7', 'recall_7', 'f1_7', 'auc_7', 'gm_7', 'bpp_7', 'mcc_7',
                 'precision_8', 'recall_8', 'f1_8', 'auc_8', 'gm_8', 'bpp_8', 'mcc_8',
                 'precision_9', 'recall_9', 'f1_9', 'auc_9', 'gm_9', 'bpp_9', 'mcc_9'], dtype=object)

    for root, dirs, files in os.walk(working_dir):

        for name in files:

            # df_name for spv
            df_name = pd.read_csv(working_dir + name)
            print("The current releases respectively are ", name)

            print(df_name.columns.values.tolist())
            metrics = df_name.columns.values.tolist()

            # bugBinary表示bug的二进制形式
            df_name["bugBinary"] = df_name.bug.apply(lambda x: 1 if x > 0 else 0)

            # pred_20 存储20个度量应用中位数阈值比较之后的得分
            df_name['pred_22'] = 0

            for metric in metrics:

                if metric == 'bug' or metric == 'File':
                    continue

                print(metric, name[-11:], df_name[metric].sum(), len(set(df_name[metric].values)))
                # 全为零的过滤
                if df_name[metric].sum() == 0 and len(set(df_name[metric].values)) == 1:
                    continue

                print("the current file is ", name, "the current metric is ", metric)
                df_name = df_name[~df_name[metric].isin(['undef', 'undefined'])].reset_index(drop=True)
                df_name = df_name[~df_name[metric].isnull()].reset_index(drop=True)
                if name[:-4] in ['Apache', 'Safe', 'zxing']:
                    df_name = df_name[~df_name['AvgLineCode'].isin([0])].reset_index(drop=True)
                else:
                    df_name = df_name[~df_name['size'].isin([0])].reset_index(drop=True)

                metric_t = df_name[metric].median()
                print("version,  metric, and its threshold value are ", name, metric, metric_t)

                # pred_22用于存储22个通用度量预测的得分，最大值为22，全预测为有缺陷，最小值为0，全预测为无缺陷。
                # 此处假设所有度量都与度量正相关。
                df_name['pred_22'] = df_name.apply(
                    lambda x: x['pred_22'] + 1 if float(x[metric]) >= metric_t else x['pred_22'] + 0, axis=1)

            # pred_22_score用于存储20个通用度量预测的得分再加上小数部分，小数部分等于当前模块的SLOC的倒数。
            # if name[-11:] == '_golden.csv':
            #     df_name['pred_22_score'] = df_name.apply(lambda x: x['pred_22'] + (1 / x['AvgLineCode']), axis=1)
            # else:
            #     df_name['pred_22_score'] = df_name.apply(lambda x: x['pred_22'] + (1 / x['size']), axis=1)

            # 应用pred_22_score的投票得分，计算CE值，再用0.1,0.2,...,0.9九种阈值分别计算分类性能
            precision_1, recall_1, f1_1, auc_1, gm_1, bpp_1, mcc_1, precision_2, recall_2, f1_2, auc_2, gm_2, bpp_2, mcc_2, \
            precision_3, recall_3, f1_3, auc_3, gm_3, bpp_3, mcc_3, precision_4, recall_4, f1_4, auc_4, gm_4, bpp_4, mcc_4, \
            precision_5, recall_5, f1_5, auc_5, gm_5, bpp_5, mcc_5, precision_6, recall_6, f1_6, auc_6, gm_6, bpp_6, mcc_6, \
            precision_7, recall_7, f1_7, auc_7, gm_7, bpp_7, mcc_7, precision_8, recall_8, f1_8, auc_8, gm_8, bpp_8, mcc_8, \
            precision_9, recall_9, f1_9, auc_9, gm_9, bpp_9, mcc_9 = predict_score_performance_precision(df_name)

            df_median = df_median.append(
                {'project': name.split('_')[0], 'current_version': name[:-4], 'Sample_size': len(df_name),
                 'precision_1': precision_1, 'recall_1': recall_1, 'f1_1': f1_1, 'auc_1': auc_1, 'gm_1': gm_1,
                 'bpp_1': bpp_1, 'mcc_1': mcc_1, 'precision_2': precision_2, 'recall_2': recall_2, 'f1_2': f1_2,
                 'auc_2': auc_2, 'gm_2': gm_2, 'bpp_2': bpp_2, 'mcc_2': mcc_2, 'precision_3': precision_3,
                 'recall_3': recall_3, 'f1_3': f1_3, 'auc_3': auc_3, 'gm_3': gm_3, 'bpp_3': bpp_3, 'mcc_3': mcc_3,
                 'precision_4': precision_4, 'recall_4': recall_4, 'f1_4': f1_4, 'auc_4': auc_4, 'gm_4': gm_4,
                 'bpp_4': bpp_4, 'mcc_4': mcc_4, 'precision_5': precision_5, 'recall_5': recall_5, 'f1_5': f1_5,
                 'auc_5': auc_5, 'gm_5': gm_5, 'bpp_5': bpp_5, 'mcc_5': mcc_5, 'precision_6': precision_6,
                 'recall_6': recall_6, 'f1_6': f1_6, 'auc_6': auc_6, 'gm_6': gm_6, 'bpp_6': bpp_6, 'mcc_6': mcc_6,
                 'precision_7': precision_7, 'recall_7': recall_7, 'f1_7': f1_7, 'auc_7': auc_7, 'gm_7': gm_7,
                 'bpp_7': bpp_7, 'mcc_7': mcc_7, 'precision_8': precision_8, 'recall_8': recall_8, 'f1_8': f1_8,
                 'auc_8': auc_8, 'gm_8': gm_8, 'bpp_8': bpp_8, 'mcc_8': mcc_8, 'precision_9': precision_9,
                 'recall_9': recall_9, 'f1_9': f1_9, 'auc_9': auc_9, 'gm_9': gm_9, 'bpp_9': bpp_9, 'mcc_9': mcc_9},
                ignore_index=True)

            df_median.to_csv(result_dir + 'median_threshold_all_versions.csv', index=False)



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

    work_Directory = "F:/PERCEIVER/Nam_data_preprocessed/"
    result_Directory = "F:/PERCEIVER/cla_on_nam_data/"

    print(os.getcwd())
    os.chdir(work_Directory)
    print(work_Directory)
    print(os.getcwd())

    median_threshold_on_current_version(work_Directory, result_Directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")
