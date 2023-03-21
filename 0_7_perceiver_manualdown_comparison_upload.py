#!/usr/bin/env python
# encoding:utf-8
"""
Project: Object-oriented-Metric-Thresholds
File: 0_7
Date : 2023/1/31 23:03

在Apache数据集65版本上比较MANULADOWN与PERCEIVER方法的性能
"""


def do_wilcoxon(x, y):
    '''
    Calculate the Wilcoxon signed-rank test.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
The Wilcoxon signed-rank test tests the null hypothesis that two related paired samples come from the same distribution.
    :param x:
    :param y:
    :return:  the p value and the cliff estimation
    '''
    from scipy import stats
    import rpy2.robjects as robjects
    import pandas as pd
    import numpy as np

    # display all columns and rows, and set the item of row of dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    robjects.r('''
    library("rcompanion")
    library("effsize")
               ''')

    if len(x) < 3 or x == y:
        wilcoxon = '/'
        cliff_estimate = '/'
    else:
        wilcoxon = stats.wilcoxon(x, y).pvalue
        x_name = ['clami_meta'] * len(x)
        y_name = ['meta'] * len(y)
        cliff = robjects.r['cliff.delta'](d=robjects.FloatVector(x + y), f=robjects.StrVector(x_name + y_name))
        cliff_estimate = np.array(cliff.rx('estimate')).flatten()[0]

    return wilcoxon, cliff_estimate


def do_cliff(work_dir, df_x, df_y, name_x, name_y):
    '''
    :param x: a list of performance indicators, such as auc, f1, gm, bpp
    :param y: a list of performance indicators, such as auc, f1, gm, bpp
    :return:  the file including the p value and cliff estimate of each OO metric
    '''
    from scipy import stats
    import rpy2.robjects as robjects
    import pandas as pd
    import numpy as np

    cmp_x_with_y = pd.DataFrame(
        columns=['f1_' + name_x + '_avg', 'f1_' + name_x + '_var', 'f1_' + name_y + '_avg', 'f1_' + name_y + '_var',
                 'f1_wilcoxon', 'f1_cliff',
                 'gm_' + name_x + '_avg', 'gm_' + name_x + '_var', 'gm_' + name_y + '_avg', 'gm_' + name_y + '_var',
                 'gm_wilcoxon', 'gm_cliff',
                 'bpp_' + name_x + '_avg', 'bpp_' + name_x + '_var', 'bpp_' + name_y + '_avg', 'bpp_' + name_y + '_var',
                 'bpp_wilcoxon', 'bpp_cliff',
                 'mcc_' + name_x + '_avg', 'mcc_' + name_x + '_var', 'mcc_' + name_y + '_avg', 'mcc_' + name_y + '_var',
                 'mcc_wilcoxon', 'mcc_cliff'])

    x_f1, y_f1 = [], []
    x_gm, y_gm = [], []
    x_bpp, y_bpp = [], []
    x_mcc, y_mcc = [], []

    for fileName in df_x['current_version'].unique():

        if fileName in df_y['version'].unique():

            x_f1.append(df_x[df_x['current_version'] == fileName].loc[:, 'f1_6'].tolist()[0])
            x_gm.append(df_x[df_x['current_version'] == fileName].loc[:, 'gm_6'].tolist()[0])
            x_bpp.append(df_x[df_x['current_version'] == fileName].loc[:, 'bpp_6'].tolist()[0])
            x_mcc.append(df_x[df_x['current_version'] == fileName].loc[:, 'mcc_6'].tolist()[0])

            y_f1.append(df_y[df_y['version'] == fileName].loc[:, 'F1'].tolist()[0])
            y_gm.append(df_y[df_y['version'] == fileName].loc[:, 'GM'].tolist()[0])
            y_bpp.append(df_y[df_y['version'] == fileName].loc[:, 'BPP'].tolist()[0])
            y_mcc.append(df_y[df_y['version'] == fileName].loc[:, 'MCC'].tolist()[0])

    f1_wilcoxon, f1_cliff_estimate = do_wilcoxon(x_f1, y_f1)
    gm_wilcoxon, gm_cliff_estimate = do_wilcoxon(x_gm, y_gm)
    bpp_wilcoxon, bpp_cliff_estimate = do_wilcoxon(x_bpp, y_bpp)
    mcc_wilcoxon, mcc_cliff_estimate = do_wilcoxon(x_mcc, y_mcc)

    cmp_x_with_y = cmp_x_with_y.append(
        {'f1_' + name_x + '_avg': np.mean(x_f1), 'f1_' + name_x + '_var': np.var(x_f1),
         'f1_' + name_y + '_avg': np.mean(y_f1), 'f1_' + name_y + '_var': np.var(y_f1),
         'f1_wilcoxon': f1_wilcoxon, 'f1_cliff': f1_cliff_estimate,
         'gm_' + name_x + '_avg': np.mean(x_gm), 'gm_' + name_x + '_var': np.var(x_gm),
         'gm_' + name_y + '_avg': np.mean(y_gm), 'gm_' + name_y + '_var': np.var(y_gm),
         'gm_wilcoxon': gm_wilcoxon, 'gm_cliff': gm_cliff_estimate,
         'bpp_' + name_x + '_avg': np.mean(x_bpp), 'bpp_' + name_x + '_var': np.var(x_bpp),
         'bpp_' + name_y + '_avg': np.mean(y_bpp), 'bpp_' + name_y + '_var': np.var(y_bpp),
         'bpp_wilcoxon': bpp_wilcoxon, 'bpp_cliff': bpp_cliff_estimate,
         'mcc_' + name_x + '_avg': np.mean(x_mcc), 'mcc_' + name_x + '_var': np.var(x_mcc),
         'mcc_' + name_y + '_avg': np.mean(y_mcc), 'mcc_' + name_y + '_var': np.var(y_mcc),
         'mcc_wilcoxon': mcc_wilcoxon, 'mcc_cliff': mcc_cliff_estimate}, ignore_index=True)

    cmp_x_with_y.to_csv(work_dir + 'cmp_' + name_x + '_0.6_with_' + name_y + '.csv', index=False)

    return cmp_x_with_y


def manualdown_perceiver_comparison(working_dir, result_dir):
    # display all columns and rows, and set the item of row of dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    # display all columns and rows, and set the item of row of dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    manualdown_dir = 'F:/dissertation/chapter_4_RQ3/'

    perceiver_dir = working_dir + 'cla_sloc_on_62_metrics_data_6_indicators/'
    # perceiver_dir = working_dir + 'perceiver_on_62_metrics_data/'

    df_manualdown = pd.read_csv(manualdown_dir + 'manualdown_all_versions.csv')
    # df_perceiver = pd.read_csv(perceiver_dir + 'median_threshold_on_65_releases.csv')
    df_perceiver = pd.read_csv(perceiver_dir + 'median_threshold_all_versions.csv')
    # df_perceiver = pd.read_csv(perceiver_dir + 'gm_threshold_on_65_releases.csv')
    # df_perceiver = pd.read_csv(perceiver_dir + 'gm_threshold_all_versions.csv')

    print(df_perceiver.head())
    print(df_manualdown.head())

    # do_cliff(result_dir, df_perceiver, df_manualdown, 'perceiver_2', 'manualdown')
    do_cliff(result_dir, df_perceiver, df_manualdown, 'perceiver_1', 'manualdown')


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

    work_Directory = "F:/PERCEIVER/"

    result_Directory = "F:/PERCEIVER/perceiver_versus_manualdown_62_metrics_data_comparison/"

    print(os.getcwd())
    os.chdir(work_Directory)
    print(work_Directory)
    print(os.getcwd())

    manualdown_perceiver_comparison(work_Directory, result_Directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")