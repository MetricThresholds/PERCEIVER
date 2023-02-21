#!/usr/bin/env python
# encoding:utf-8
"""
Project: Object-oriented-Metric-Thresholds
File: 0_3_cla_perceiver_comparison.py
Date : 2022/12/14 21:49

compare perceiver wiht baselines
将PERCEIVER方法与基线方法比较

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


def cla_perceiver_comparison(working_dir, result_dir):
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

    cla_dir = working_dir + 'cla_on_62_metrics_data_6_indicators/'

    perceiver_dir = working_dir + 'perceiver_on_62_metrics_data_6_indicators/'

    df_cla = pd.read_csv(cla_dir + 'median_threshold_on_65_releases.csv')

    df_perceiver = pd.read_csv(perceiver_dir + 'gm_threshold_on_65_releases.csv')

    cmp_cla_with_perceiver = pd.DataFrame(columns=['performance', 'avg_perceiver', 'avg_cla', 'wilcoxon', 'cliff'])

    for column in df_cla.columns.values:

        if column in ['project', 'current_version', 'Sample_size']:
            continue

        cla_values, perceiver_values = [], []

        for project in df_cla['project'].unique():

            if project in df_perceiver['project'].unique():
                cla_values.append(df_cla[df_cla['project'] == project].loc[:, column].tolist()[0])
                perceiver_values.append(df_perceiver[df_perceiver['project'] == project].loc[:, column].tolist()[0])

        wilcoxon, liff_estimate = do_wilcoxon(perceiver_values, cla_values)

        cmp_cla_with_perceiver = cmp_cla_with_perceiver.append(
            {'performance': column, 'avg_perceiver': np.mean(perceiver_values), 'avg_cla': np.mean(cla_values),
             'wilcoxon': wilcoxon, 'cliff': liff_estimate}, ignore_index=True)

        cmp_cla_with_perceiver.to_csv(result_dir + 'cmp_percever_versus_cla_on_62_metrics_data.csv', index=False)


if __name__ == '__main__':
    import os
    import sys
    import time
    import pandas as pd
    import numpy as np

    s_time = time.time()

    work_Directory = "F:/PERCEIVER/"

    result_Directory = "F:/PERCEIVER/perceiver_on_62_metrics_data_6_indicators/perceiver_versus_cla_on_62_metrics_data_comparison/"

    os.chdir(work_Directory)

    cla_perceiver_comparison(work_Directory, result_Directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")
