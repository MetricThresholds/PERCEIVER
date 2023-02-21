#!/usr/bin/env python
# encoding:utf-8
"""
Project: Object-oriented-Metric-Thresholds
File: 5_1_perceiver_meta_on_65_releases.py
Date : 2023/2/2 16:33

由于chapter_4_0_2_voting_meta_threshold.py
参考chapter_4_0_0_unsupervised_median_thrshold.py方法
参考ActiveLearningThrehsold项目中4_3_unsupervised_median_threshold.py
元分析阈值的投票方法： 用当前版本上所有度量与其元分析阈值比较，得到评分后，再加上SLOC度量的倒数，得到当前版本上所有模块的综合评分，
           取50%的综合评分中位数阈值预测各模块缺陷倾向。
数据集为第三章元分析中测试集65个版本。

"""


# 分别应用pred_22_score的投票得分，用0.1,0.2,...,0.9 九种对应百分比分位数作为阈值分别计算分类性能
def do_predict(df_test):
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


def meta_threshold_on_current_version(working_dir, result_dir):
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

    dir_data = working_dir + 'data_defects_java_testing/'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # 22 object-oriented features and a dependent variable:  'bug'
    metrics_22 = ['LCOM1', 'LCOM2', 'LCOM3', 'ICH', 'NHD', 'OCAIC', 'OCMIC', 'OMMIC', 'CBO', 'DAC', 'ICP', 'MPC',
                  'NIHICP', 'RFC', "NMA", "NA", "NAIMP", "NM", "NMIMP", "NumPara", "SLOC", "stms"]

    metrics_26 = ['LCOM1', 'LCOM2', 'LCOM3', 'LCOM5', 'NewLCOM5', 'ICH', 'CAMC', 'NHD', 'SNHD', 'OCAIC', 'OCMIC',
                  'OMMIC', 'CBO', 'DAC', 'ICP', 'MPC', 'NIHICP', 'RFC', "NMA", "NA", "NAIMP", "NM", "NMIMP", "NumPara",
                  "SLOC", "stms"]

    metrics_3_negative = ['NewLCOM5', 'CAMC', 'SNHD']

    # store predictive performance of all versions
    df_voting_meta = pd.DataFrame(
        columns=['project', 'current_version', 'Sample_size', 'f1_0.1', 'gm_0.1', 'bpp_0.1', 'mcc_0.1',
                 'f1_0.2', 'gm_0.2', 'bpp_0.2', 'mcc_0.2', 'f1_0.3', 'gm_0.3', 'bpp_0.3', 'mcc_0.3',
                 'f1_0.4', 'gm_0.4', 'bpp_0.4', 'mcc_0.4', 'f1_0.5', 'gm_0.5', 'bpp_0.5', 'mcc_0.5',
                 'f1_0.6', 'gm_0.6', 'bpp_0.6', 'mcc_0.6', 'f1_0.7', 'gm_0.7', 'bpp_0.7', 'mcc_0.7',
                 'f1_0.8', 'gm_0.8', 'bpp_0.8', 'mcc_0.8', 'f1_0.9', 'gm_0.9', 'bpp_0.9', 'mcc_0.9'], dtype=object)

    # 读出元分析阈值
    df_meta = pd.read_csv("F:/dissertation/all_universalThresholds.csv", keep_default_na=False, na_values=[""])

    for root, dirs, files in os.walk(dir_data):

        for name in files:

            # df_name for spv
            df_name = pd.read_csv(dir_data + name)
            print("The current releases respectively are ", name)
            # bugBinary表示bug的二进制形式
            df_name["bugBinary"] = df_name.bug.apply(lambda x: 1 if x > 0 else 0)

            # pred_20 存储20个度量应用中位数阈值比较之后的得分
            df_name['pred_22'] = 0

            for metric in metrics_26:
                print("the current file is ", name, "the current metric is ", metric)
                df_name = df_name[~df_name[metric].isin(['undef', 'undefined'])].reset_index(drop=True)
                df_name = df_name[~df_name[metric].isnull()].reset_index(drop=True)
                df_name = df_name[~df_name['SLOC'].isin([0])].reset_index(drop=True)

                metric_t = df_meta[df_meta['metric'] == metric].loc[:, 'universal_t_rounded'].tolist()[0]
                # metric_t = df_name[metric].median()
                print("version,  metric, and its threshold value are ", name, metric, metric_t)

                # pred_26用于存储26个通用度量预测的得分，最大值为26，全预测为有缺陷，最小值为0，全预测为无缺陷。
                if metric in metrics_3_negative:
                    df_name['pred_22'] = df_name.apply(
                        lambda x: x['pred_22'] + 1 if float(x[metric]) <= metric_t else x['pred_22'] + 0, axis=1)
                else:
                    df_name['pred_22'] = df_name.apply(
                        lambda x: x['pred_22'] + 1 if float(x[metric]) >= metric_t else x['pred_22'] + 0, axis=1)

            # pred_22_score用于存储20个通用度量预测的得分再加上小数部分，小数部分等于当前模块的SLOC的倒数。
            df_name['pred_22_score'] = df_name.apply(lambda x: x['pred_22'] + (1 / x['SLOC']), axis=1)

            # 应用pred_22_score的投票得分，计算CE值，再用0.1,0.2,...,0.9九种阈值分别计算分类性能
            f1_1, gm_1, bpp_1, mcc_1, f1_2, gm_2, bpp_2, mcc_2, f1_3, gm_3, bpp_3, mcc_3, f1_4, gm_4, bpp_4, mcc_4, \
            f1_5, gm_5, bpp_5, mcc_5, f1_6, gm_6, bpp_6, mcc_6, f1_7, gm_7, bpp_7, mcc_7, f1_8, gm_8, bpp_8, mcc_8, \
            f1_9, gm_9, bpp_9, mcc_9 = do_predict(df_name)

            df_voting_meta = df_voting_meta.append(
                {'project': name.split('-')[0], 'current_version': name[:-4], 'Sample_size': len(df_name),
                 'f1_0.1': f1_1, 'gm_0.1': gm_1, 'bpp_0.1': bpp_1, 'mcc_0.1': mcc_1, 'f1_0.2': f1_2, 'gm_0.2': gm_2,
                 'bpp_0.2': bpp_2, 'mcc_0.2': mcc_2, 'f1_0.3': f1_3, 'gm_0.3': gm_3, 'bpp_0.3': bpp_3, 'mcc_0.3': mcc_3,
                 'f1_0.4': f1_4, 'gm_0.4': gm_4, 'bpp_0.4': bpp_4, 'mcc_0.4': mcc_4, 'f1_0.5': f1_5, 'gm_0.5': gm_5,
                 'bpp_0.5': bpp_5, 'mcc_0.5': mcc_5, 'f1_0.6': f1_6, 'gm_0.6': gm_6, 'bpp_0.6': bpp_6, 'mcc_0.6': mcc_6,
                 'f1_0.7': f1_7, 'gm_0.7': gm_7, 'bpp_0.7': bpp_7, 'mcc_0.7': mcc_7, 'f1_0.8': f1_8, 'gm_0.8': gm_8,
                 'bpp_0.8': bpp_8, 'mcc_0.8': mcc_8, 'f1_0.9': f1_9, 'gm_0.9': gm_9, 'bpp_0.9': bpp_9,
                 'mcc_0.9': mcc_9}, ignore_index=True)

            df_voting_meta.to_csv(result_dir + 'meta_threshold_all_versions.csv', index=False)



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
    result_Directory = "F:/PERCEIVER/PERCEIVER_meta/"

    print(os.getcwd())
    os.chdir(work_Directory)
    print(work_Directory)
    print(os.getcwd())

    meta_threshold_on_current_version(work_Directory, result_Directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")