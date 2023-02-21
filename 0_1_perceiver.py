#!/usr/bin/env python
# encoding:utf-8
"""
Project: Object-oriented-Metric-Thresholds
File: 0_1_perceiver.py
Date : 2022/12/14 11:33


基本方法如下：
（1）用中位数阈值投票，得到伪标签
（2）在伪标签计算每个度量GM最大化下的阈值，然后将计算阈值过程中的GM值按从大到小排序，舍去后半预测性能差的度量
（3）在剩下的前半部分度量上，分别试两种方法投票得分，一是GM最大化阈值下的投票得分，二是还用中位数投票，并与之前（1）中的性能做比较。


"""



# output: the threshold derived from MGM
# note that the dataframe should input astype(float), i.e., MGM_threshold(df.astype(float))
def max_gm_threshold(df):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix

    metric_name = ''
    for col in df.columns:
        if col not in ['bug', 'pseudo_bug']:
            metric_name = col

    # 2. 依次用该度量每一个值作为阈值计算出各自预测性能值,然后选择预测性能值最大的作为阈值,分别定义存入list,取最大值和最大值的下标值
    # 同时输出gm最大值阈值的F1和BPP值。
    GMs = []
    gm_max_value = 0
    f1_with_gm_max = 0
    bpp_with_gm_max = 0
    i_gm_max = 0

    # 判断每个度量与bug之间的关系,用于阈值判断正反例
    Corr_metric_bug = df.loc[:, [metric_name, 'pseudo_bug']].corr('spearman')

    Spearman_value = Corr_metric_bug[metric_name][1]
    Pearson_value = 2 * np.sin(np.pi * Spearman_value / 6)

    # the i value in this loop, is the subscript value in the list of AUCs, GMs etc.
    for i in range(len(df)):

        t = df.loc[i, metric_name]

        if Pearson_value < 0:
            df['predictBinary'] = df[metric_name].apply(lambda x: 1 if x <= t else 0)
        else:
            df['predictBinary'] = df[metric_name].apply(lambda x: 1 if x >= t else 0)

        # confusion_matrix()函数中需要给出label, 0和1，否则该函数算不出TP,因为不知道哪个标签是poistive.
        c_matrix = confusion_matrix(df["pseudo_bug"], df['predictBinary'], labels=[0, 1])
        tn, fp, fn, tp = c_matrix.ravel()

        if (tn + fp) == 0:
            tnr_value = 0
        else:
            tnr_value = tn / (tn + fp)

        if (fp + tn) == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)

        # auc_value = roc_auc_score(df['pseudo_bug'], df['predictBinary'])
        recall_value = recall_score(df['pseudo_bug'], df['predictBinary'], labels=[0, 1])
        # precision_value = precision_score(df['bugBinary'], df['predictBinary'], labels=[0, 1])
        f1_value = f1_score(df['pseudo_bug'], df['predictBinary'], labels=[0, 1])

        gm_value = (recall_value * tnr_value) ** 0.5
        pdr = recall_value
        pfr = fpr  # fp / (fp + tn)
        bpp_value = 1 - (((0 - pfr) ** 2 + (1 - pdr) ** 2) * 0.5) ** 0.5

        GMs.append(gm_value)

        # 求出上述list中最大值，及对应的i值，可能会有几个值相同，且为最大值，则取第一次找到那个值(i)为阈值
        if gm_value > gm_max_value:
            gm_max_value = gm_value
            f1_with_gm_max = f1_value
            bpp_with_gm_max = bpp_value
            i_gm_max = i

    # 计算阈值,包括其他四个类型阈值
    gm_t = df.loc[i_gm_max, metric_name]

    return Pearson_value, gm_t, gm_max_value, f1_with_gm_max, bpp_with_gm_max


# 分别应用pred_22_score的投票得分，用0.1,0.2,...,0.9 九种对应百分比分位数作为阈值分别计算分类性能
def do_predict(df_test):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
    f1_1, gm_1, bpp_1, f1_2, gm_2, bpp_2, f1_3, gm_3, bpp_3, f1_4, gm_4, bpp_4, f1_5, gm_5, bpp_5, \
    f1_6, gm_6, bpp_6, f1_7, gm_7, bpp_7, f1_8, gm_8, bpp_8, f1_9, gm_9, bpp_9 = \
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for i in range(9):

        df_predict = df_test.copy()

        for j in range(len(df_predict)):
            if df_predict.loc[j, 'pseudo_pred_22_score'] >= df_predict.pseudo_pred_22_score.quantile(0.1 * (i + 1)):
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


# 分别应用pred_22_score的投票得分，用0.1,0.2,...,0.9 九种对应百分比分位数作为阈值分别计算分类性能
def do_predict_gm(df_test):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
    f1_1, gm_1, bpp_1, f1_2, gm_2, bpp_2, f1_3, gm_3, bpp_3, f1_4, gm_4, bpp_4, f1_5, gm_5, bpp_5, \
    f1_6, gm_6, bpp_6, f1_7, gm_7, bpp_7, f1_8, gm_8, bpp_8, f1_9, gm_9, bpp_9 = \
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for i in range(9):

        df_predict = df_test.copy()

        for j in range(len(df_predict)):
            if df_predict.loc[j, 'pseudo_pred_22_score_gm'] >= df_predict.pseudo_pred_22_score_gm.quantile(0.1 * (i + 1)):
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


def perceiver_on_current_version(working_dir, result_dir):
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

    # df_perceiver_median 存储删去一半度量还用中位数阈值的结果
    df_perceiver_median = pd.DataFrame(
        columns=['project', 'current_version', 'Sample_size', 'f1_0.1', 'gm_0.1', 'bpp_0.1',
                 'f1_0.2', 'gm_0.2', 'bpp_0.2', 'f1_0.3', 'gm_0.3', 'bpp_0.3', 'f1_0.4', 'gm_0.4', 'bpp_0.4',
                 'f1_0.5', 'gm_0.5', 'bpp_0.5', 'f1_0.6', 'gm_0.6', 'bpp_0.6', 'f1_0.7', 'gm_0.7', 'bpp_0.7',
                 'f1_0.8', 'gm_0.8', 'bpp_0.8', 'f1_0.9', 'gm_0.9', 'bpp_0.9'], dtype=object)

    # df_perceiver_gm 存储删去一半度量用在伪标签上的GM最大化的阈值的结果
    df_perceiver_gm = pd.DataFrame(
        columns=['project', 'current_version', 'Sample_size', 'f1_0.1', 'gm_0.1', 'bpp_0.1',
                 'f1_0.2', 'gm_0.2', 'bpp_0.2', 'f1_0.3', 'gm_0.3', 'bpp_0.3', 'f1_0.4', 'gm_0.4', 'bpp_0.4',
                 'f1_0.5', 'gm_0.5', 'bpp_0.5', 'f1_0.6', 'gm_0.6', 'bpp_0.6', 'f1_0.7', 'gm_0.7', 'bpp_0.7',
                 'f1_0.8', 'gm_0.8', 'bpp_0.8', 'f1_0.9', 'gm_0.9', 'bpp_0.9'], dtype=object)

    for root, dirs, files in os.walk(dir_data):

        for name in files:

            # df_name for spv
            df_name = pd.read_csv(dir_data + name)
            print("The current releases respectively are ", name)

            df_name["bugBinary"] = df_name.bug.apply(lambda x: 1 if x > 0 else 0)

            # pred_22 存储22个度量应用中位数阈值比较之后的得分
            df_name['pred_22'] = 0
            df_name['pseudo_bug'] = 0

            # pseudo_pred_22存储删一半度量后的用中位数阈值预测值
            df_name['pseudo_pred_22'] = 0

            # pseudo_pred_22存储删一半度量后的用gm最大化阈值预测值
            df_name['pseudo_pred_22_gm'] = 0

            # to_be_deleted_metrics用于存储一半被删去的度量
            to_be_deleted_metrics = []

            # 计算伪标签用于训练阈值
            for metric in metrics_22:
                print("the current file is ", name, "the current metric is ", metric)
                df_name = df_name[~df_name[metric].isin(['undef', 'undefined'])].reset_index(drop=True)
                df_name = df_name[~df_name[metric].isnull()].reset_index(drop=True)
                df_name = df_name[~df_name['SLOC'].isin([0])].reset_index(drop=True)

                metric_t = df_name[metric].median()
                print("version,  metric, and its threshold value are ", name, metric, metric_t)

                # pred_22用于存储22个通用度量预测的得分，最大值为22，全预测为有缺陷，最小值为0，全预测为无缺陷。
                # 此处假设所有度量都与度量正相关。
                df_name['pred_22'] = df_name.apply(
                    lambda x: x['pred_22'] + 1 if float(x[metric]) >= metric_t else x['pred_22'] + 0, axis=1)

            # pred_22_score用于存储20个通用度量预测的得分再加上小数部分，小数部分等于当前模块的SLOC的倒数。
            df_name['pred_22_score'] = df_name.apply(lambda x: x['pred_22'] + (1 / x['SLOC']), axis=1)
            df_name['pseudo_bug'] = df_name.apply(
                lambda x: 1 if float(x['pred_22_score']) >= df_name['pred_22_score'].median() else 0, axis=1)

            print(df_name['pred_22_score'].median())
            # print(df_name.loc[:, ['pred_22_score', 'pseudo_bug']])

            # 各度量的阈值,iter_gm_max用于存储计算该阈值的GM最大值，然后删去GM最大值从在大到小，后半数量的度量
            iter_t, iter_pearson, iter_gm_max, iter_f1_with_gm_max, iter_bpp_with_gm_max = {}, {}, {}, {}, {}

            # 计算阈值，然后删去计算阈值过程中GM最大值从大到小的后一半度量
            for metric in metrics_22:

                pearson_t0, gm_t0, gm_max, f1_with_gm_max, bpp_with_gm_max = \
                    max_gm_threshold(df_name.loc[:, ['pseudo_bug', metric]].astype(float))
                iter_t[metric] = gm_t0
                iter_pearson[metric] = pearson_t0
                iter_gm_max[metric] = gm_max
                iter_f1_with_gm_max[metric] = f1_with_gm_max
                iter_bpp_with_gm_max[metric] = bpp_with_gm_max
                print("the current file, metric, threshold, gm_max are ", name, metric, gm_t0, gm_max)

            print(iter_gm_max)
            print(sorted(iter_gm_max))
            print(iter_gm_max)

            # to_be_deleted_metrics中最多保留一半度量
            gm_max_list = {}
            # print(iter_gm_max.values())
            for key, value in iter_gm_max.items():
                # print(key, value)
                if value != '/':
                    gm_max_list[key] = value
            print(gm_max_list)
            print(gm_max_list.values())
            print(type(gm_max_list.values()))
            print(repr(gm_max_list.values()))
            print(repr(list(gm_max_list.values())))
            print(np.median(list(gm_max_list.values())))
            for key, value in gm_max_list.items():
                if (value <= np.median(list(gm_max_list.values()))):
                    print("the metric and it gm value which is the minimum value ", key, value)
                    to_be_deleted_metrics.append(key)

            print(to_be_deleted_metrics)

            # df_perceiver_median 存储删去一半度量还用中位数阈值的结果
            for metric in metrics_22:

                # 若该度量在上一轮迭代中被删，则不参与投票
                if metric in to_be_deleted_metrics:
                    continue

                metric_t = df_name[metric].median()
                metric_t_gm = iter_t[metric]
                print("version,  metric, and its threshold value are ", name, metric, metric_t)

                # pred_22用于存储22个通用度量预测的得分，最大值为22，全预测为有缺陷，最小值为0，全预测为无缺陷。
                # 此处假设所有度量都与度量正相关。
                df_name['pseudo_pred_22'] = df_name.apply(
                    lambda x: x['pseudo_pred_22'] + 1 if float(x[metric]) >= metric_t else x['pseudo_pred_22'] + 0, axis=1)

                df_name['pseudo_pred_22_gm'] = df_name.apply(
                    lambda x: x['pseudo_pred_22_gm'] + 1 if float(x[metric]) >= metric_t_gm else x['pseudo_pred_22_gm'] + 0, axis=1)

            df_name['pseudo_pred_22_score'] = df_name.apply(lambda x: x['pseudo_pred_22'] + (1 / x['SLOC']), axis=1)

            # 应用pred_22_score的投票得分，计算CE值，再用0.1,0.2,...,0.9九种阈值分别计算分类性能
            f1_1, gm_1, bpp_1, f1_2, gm_2, bpp_2, f1_3, gm_3, bpp_3, f1_4, gm_4, bpp_4, f1_5, gm_5, bpp_5, \
            f1_6, gm_6, bpp_6, f1_7, gm_7, bpp_7, f1_8, gm_8, bpp_8, f1_9, gm_9, bpp_9 = do_predict(df_name)

            df_perceiver_median = df_perceiver_median.append(
                {'project': name.split('-')[0], 'current_version': name[:-4], 'Sample_size': len(df_name),
                 'f1_0.1': f1_1, 'gm_0.1': gm_1, 'bpp_0.1': bpp_1, 'f1_0.2': f1_2, 'gm_0.2': gm_2, 'bpp_0.2': bpp_2,
                 'f1_0.3': f1_3, 'gm_0.3': gm_3, 'bpp_0.3': bpp_3, 'f1_0.4': f1_4, 'gm_0.4': gm_4, 'bpp_0.4': bpp_4,
                 'f1_0.5': f1_5, 'gm_0.5': gm_5, 'bpp_0.5': bpp_5, 'f1_0.6': f1_6, 'gm_0.6': gm_6, 'bpp_0.6': bpp_6,
                 'f1_0.7': f1_7, 'gm_0.7': gm_7, 'bpp_0.7': bpp_7, 'f1_0.8': f1_8, 'gm_0.8': gm_8, 'bpp_0.8': bpp_8,
                 'f1_0.9': f1_9, 'gm_0.9': gm_9, 'bpp_0.9': bpp_9}, ignore_index=True)

            df_perceiver_median.to_csv(result_dir + 'median_threshold_all_versions.csv', index=False)

            df_name['pseudo_pred_22_score_gm'] = df_name.apply(lambda x: x['pseudo_pred_22_gm'] + (1 / x['SLOC']), axis=1)

            # 应用pred_22_score的投票得分，计算CE值，再用0.1,0.2,...,0.9九种阈值分别计算分类性能
            f1_1_gm, gm_1_gm, bpp_1_gm, f1_2_gm, gm_2_gm, bpp_2_gm, f1_3_gm, gm_3_gm, bpp_3_gm, f1_4_gm, gm_4_gm,\
            bpp_4_gm, f1_5_gm, gm_5_gm, bpp_5_gm, f1_6_gm, gm_6_gm, bpp_6_gm, f1_7_gm, gm_7_gm, bpp_7_gm, f1_8_gm,\
            gm_8_gm, bpp_8_gm, f1_9_gm, gm_9_gm, bpp_9_gm = do_predict_gm(df_name)

            df_perceiver_gm = df_perceiver_gm.append(
                {'project': name.split('-')[0], 'current_version': name[:-4], 'Sample_size': len(df_name),
                 'f1_0.1': f1_1_gm, 'gm_0.1': gm_1_gm, 'bpp_0.1': bpp_1_gm, 'f1_0.2': f1_2_gm, 'gm_0.2': gm_2_gm,
                 'bpp_0.2': bpp_2_gm, 'f1_0.3': f1_3_gm, 'gm_0.3': gm_3_gm, 'bpp_0.3': bpp_3_gm, 'f1_0.4': f1_4_gm,
                 'gm_0.4': gm_4_gm, 'bpp_0.4': bpp_4_gm, 'f1_0.5': f1_5_gm, 'gm_0.5': gm_5_gm, 'bpp_0.5': bpp_5_gm,
                 'f1_0.6': f1_6_gm, 'gm_0.6': gm_6_gm, 'bpp_0.6': bpp_6_gm, 'f1_0.7': f1_7_gm, 'gm_0.7': gm_7_gm,
                 'bpp_0.7': bpp_7_gm, 'f1_0.8': f1_8_gm, 'gm_0.8': gm_8_gm, 'bpp_0.8': bpp_8_gm,
                 'f1_0.9': f1_9_gm, 'gm_0.9': gm_9_gm, 'bpp_0.9': bpp_9_gm}, ignore_index=True)

            df_perceiver_gm.to_csv(result_dir + 'gm_threshold_all_versions.csv', index=False)

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
    result_Directory = "F:/PERCEIVER/perceiver/"

    print(os.getcwd())
    os.chdir(work_Directory)
    print(work_Directory)
    print(os.getcwd())

    perceiver_on_current_version(work_Directory, result_Directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")
