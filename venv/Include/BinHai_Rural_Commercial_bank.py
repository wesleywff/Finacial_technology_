#!/usr/bin/env Python
# coding=utf-8
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import toad
from sklearn.ensemble import RandomForestRegressor
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
from sklearn.impute import SimpleImputer
import time,datetime
from sklearn.model_selection import train_test_split
import variable_bin_methods as varbin_meth ## 自定义函数，已上传至GitHub同文件夹
import variable_encode as var_encode ## 自定义函数，已上传至GitHub同文件夹
from feature_selector import FeatureSelector ## 自定义函数，已上传至GitHub同文件夹
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc,confusion_matrix,recall_score,precision_score,accuracy_score





#**************************************读取数据**************************************************************************

# 每个变量缺失率的计算
def missing_cal(df):
    """
    df :数据集

    return：每个变量的缺失率
    """
    missing_series = df.isnull().sum() / df.shape[0]
    missing_df = pd.DataFrame(missing_series).reset_index()
    missing_df = missing_df.rename(columns={'index': 'col',
                                            0: 'missing_pct'})
    missing_df = missing_df.sort_values('missing_pct', ascending=False).reset_index(drop=True)
    return missing_df


# 缺失值剔除（单个变量）
def missing_delete_var(df, threshold=None):
    """
    df:数据集
    threshold:缺失率删除的阈值

    return :删除缺失后的数据集
    """
    df2 = df.copy()
    missing_df = missing_cal(df)
    missing_col_num = missing_df[missing_df.missing_pct >= threshold].shape[0]
    missing_col = list(missing_df[missing_df.missing_pct >= threshold].col)
    df2 = df2.drop(missing_col, axis=1)
    print('缺失率超过{}的变量个数为{}'.format(threshold, missing_col_num))
    return df2


# 缺失值剔除（单个样本）
def missing_delete_user(df, threshold=None):
    """
    df:数据集
    threshold:缺失个数删除的阈值

    return :删除缺失后的数据集
    """
    df2 = df.copy()
    missing_series = df.isnull().sum(axis=1)
    missing_list = list(missing_series)
    missing_index_list = []
    for i, j in enumerate(missing_list):
        if j >= threshold:
            missing_index_list.append(i)
    df2 = df2[~(df2.index.isin(missing_index_list))]
    print('缺失变量个数在{}以上的用户数有{}个'.format(threshold, len(missing_index_list)))
    return df2

## 删除常量
def constant_del(df, cols):
    dele_list = []
    for col in cols:
        # remove repeat value counts
        uniq_vals = list(df[col].unique())
        if pd.isnull(uniq_vals).any():
            if len( uniq_vals ) == 2:
                dele_list.append(col)
                print (" {} 变量只有一种取值,该变量被删除".format(col))
        elif len(df[col].unique()) == 1:
            dele_list.append(col)
            print (" {} 变量只有一种取值,该变量被删除".format(col))
    df = df.drop(dele_list, axis=1)
    return df,dele_list


# 缺失值填充（类别型变量）
def fillna_cate_var(df, col_list, fill_type=None):
    """
    df:数据集
    col_list:变量list集合
    fill_type: 填充方式：众数/当做一个类别

    return :填充后的数据集
    """
    df2 = df.copy()
    df2 = df2.reset_index(drop=False)
    for col in col_list:
        if fill_type == 'class':
            df2[col] = df2[col].fillna('unknown')
        if fill_type == 'mode':
            df2[col] = df2[col].fillna(df2[col].mode()[0])
    return df2


def missting_data_interpolation(sort_miss_index,data3):
    for i in sort_miss_index:
        data3_list = data3.columns.tolist()  # 特征名
        data3_copy = data3.copy()
        fillc = data3_copy.iloc[:, i]  # 需要填充缺失值的一列
        # 从特征矩阵中删除这列，因为要根据已有信息预测这列
        df = data3_copy.drop(data3_list[i], axis=1)
        # 将已有信息的缺失值暂用0填补
        df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)

        Ytrain = fillc[fillc.notnull()]  # 训练集标签为填充列含有数据的一部分
        Ytest = fillc[fillc.isnull()]  # 测试集标签为填充列含有缺失值的一部分

        Xtrain = df_0[Ytrain.index, :]  # 通过索引获取Xtrain和Xtest
        Xtest = df_0[Ytest.index, :]

        rfc = RandomForestRegressor(n_estimators=100)  # 实例化
        rfc = rfc.fit(Xtrain, Ytrain)  # 导入训练集进行训练
        Ypredict = rfc.predict(Xtest)  # 将Xtest传入predict方法中，得到预测结果
        # 获取原填充列中缺失值的索引
        the_index = data3[data3.iloc[:, i].isnull() == True].index.tolist()
        data3.iloc[the_index, i] = Ypredict  # 将预测好的特征填充至原始特征矩阵中


##删除长尾数据、集中度高的数据
def tail_del(df,cols,rate):
    dele_list = []
    len_1 = df.shape[0]
    for col in cols:
        if len(df[col].unique()) < 5:
            if df[col].value_counts().max()/len_1 >= rate:
                dele_list.append(col)
                print (" {} 变量分布不均衡,该变量被删除".format(col))
    df = df.drop(dele_list, axis=1)
    return df,dele_list


# 常变量/同值化处理
def const_delete(df, col_list, threshold=None):
    """
    df:数据集
    col_list:变量list集合
    threshold:同值化处理的阈值

    return :处理后的数据集
    """
    df2 = df.copy()
    const_col = []

    for col in col_list:
        const_pct = df2[col].value_counts().iloc[0] / df2[df2[col].notnull()].shape[0]
        if const_pct >= threshold:
            const_col.append(col)
    df2 = df2.drop(const_col, axis=1)
    print('常变量/同值化处理的变量个数为{}'.format(len(const_col)))
    return df2


# 分类型变量的降基处理
def descending_cate(df, col_list, threshold=None):
    """
    df: 数据集
    col_list:变量list集合
    threshold:降基处理的阈值

    return :处理后的数据集
    """
    df2 = df.copy()
    for col in col_list:
        value_series = df[col].value_counts() / df[df[col].notnull()].shape[0]
        small_value = []
        for value_name, value_pct in zip(value_series.index, value_series.values):
            if value_pct <= threshold:
                small_value.append(value_name)
        df2.loc[df2[col].isin(small_value), col] = 'other'
    return df2



## 变量选择：iv筛选
def iv_selection_func(bin_data, data_params, iv_low=0.02, iv_up=5, label='target'):
    # 简单看一下IV，太小的不要
    selected_features = []
    for k, v in data_params.items():
        if iv_low <= v < iv_up and k in bin_data.columns:
            selected_features.append(k+'_woe')
        else:
            print('{0} 变量的IV值为 {1}，小于阈值删除'.format(k, v))
    selected_features.append(label)
    return bin_data[selected_features]

def encoding_data(x):
    if pd.isnull(x):
        return x
    else:
        if "有限责任公司" in x:
            x="0"
        elif "个体工商户" in x:
            x = "2"
        elif "个人独资企业" in x:
            x = "3"
        elif  "股份" in x:
            x = "4"
        elif "普通合伙" in x:
            x = "5"
    return x

def encoding_type(x):
    if pd.isnull(x):
        return x
    else:
        if ("存续" in x) or ("在营" in x) or ("正常" in x) or ("在业" in x) or ("开业" in x):
            x = "0"
        elif ("注销" in x) or ("吊销" in x) or ("撤销" in x) or ("迁出" in x):
            x = "1"
    return x

def numeric_interpolation(data5,interpo_type,missing_col):
    x_full = data5[numeric_vars].copy()  # x.shape = (506, 13)，共有506个样本，每个样本有13个特征
    y_full = data5.target


    # 创造缺失的数据集
    x_missing = x_full.copy()

    # 数据集中缺失值从少到多进行排序
    x_missing_reg = x_missing.copy()
    x_missing_reg = x_missing_reg.reset_index(drop=True)

    # # sortindex = np.argsort(x_missing_reg.isnull().sum(axis=0)).values
    # result = pd.DataFrame(x_missing_reg.isnull().sum(axis=0))
    # results = result.loc[result.values > 0]
    k = 0

    for i in missing_col:
        print("开始%s列插补",i)
        num_col = []
        # 构建新特征矩阵和新标签
        if interpo_type == "median":
            x_missing_reg[i] = x_missing_reg[i].fillna(x_missing_reg[i].median())

        elif interpo_type == "rf":
            df = x_missing_reg
            fillc = df.loc[:, i]

            df = pd.concat([df.loc[:, df.columns != i], pd.DataFrame(y_full)], axis=1)

            # 在新特征矩阵中，对含有缺失值的列，进行0的填补
            df_0 = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0).fit_transform(df)

            # 构建新的训练集和测试集
            y_train = fillc[fillc.notnull()]
            y_test = fillc[fillc.isnull()]
            x_train = df_0[y_train.index, :]
            x_test = df_0[y_test.index, :]

            # 用随机森林填补缺失值
            rfc = RandomForestRegressor()
            rfc.fit(x_train, y_train)
            y_predict = rfc.predict(x_test)  # 用predict接口将x_test导入，得到我们的预测结果，此结果就是要用来填补空值的

            x_missing_reg.loc[x_missing_reg.loc[:, i].isnull(), i] = y_predict

            k += 1
            num = len(missing_col) - k

            print("完成{}列插补,还剩{}列需要插补".format(i,num))

    return x_missing_reg

def date_caculate(date1,date2):
    print("日期转化")
    if pd.isnull(date2):
        return date2
    else:
        print(date2)
        date2 = date2[0:10]
        star_date = datetime.datetime.strptime(date2, "%Y-%m-%d")
        date2 = date1 - star_date
    return float(date2.days)

def unit_conver(content):
    if pd.isnull(content):
        return content
    else:
        if "万元人民币" in content:
            content = content.replace("万元人民币","")
            content = float(content)
        elif "万美元" in content:
            content = content.replace("万美元","")
            content = float(content) * 6.63
    return content



## 生成评分卡
def create_score(dict_woe_map,dict_params,dict_cont_bin,dict_disc_bin):
    ##假设Odds在1:60时对应的参考分值为600分，分值调整刻度PDO为20，则计算得到分值转化的参数B = 28.85，A= 481.86。
    params_A,params_B = score_params_cal(base_point=600, odds=1/60, PDO=20)
    # 计算基础分
    base_points = round(params_A - params_B * dict_params['intercept'])
    df_score = pd.DataFrame()
    dict_bin_score = {}
    for k in dict_params.keys():
#        k='duration_BIN'
#        k = 'foreign_worker_BIN'
        if k !='intercept':
            df_temp =  pd.DataFrame([dict_woe_map[k.split(sep='_woe')[0]]]).T
            df_temp.reset_index(inplace=True)
            df_temp.columns = ['bin','woe_val']
            ## 计算分值
            df_temp['score'] = round(-params_B*df_temp.woe_val*dict_params[k])
            dict_bin_score[k.split(sep='_BIN')[0]] = dict(zip(df_temp['bin'],df_temp['score']))
            ## 连续变量的计算
            if k.split(sep='_BIN')[0] in dict_cont_bin.keys():
                df_1 = dict_cont_bin[k.split(sep='_BIN')[0]]
                df_1['var_name'] = df_1[['bin_low', 'bin_up']].apply(myfunc,axis=1)
                df_1 = df_1[['total', 'var_name']]
                df_temp = pd.merge(df_temp , df_1,on='bin')
                df_temp['var_name_raw'] = k.split(sep='_BIN')[0]
                df_score = pd.concat([df_score,df_temp],axis=0)
            ## 离散变量的计算
            elif k.split(sep='_BIN')[0] in dict_disc_bin.keys():
                df_temp = pd.merge(df_temp , dict_disc_bin[k.split(sep='_BIN')[0]],on='bin')
                df_temp['var_name_raw'] = k.split(sep='_BIN')[0]
                df_score = pd.concat([df_score,df_temp],axis=0)

    df_score['score_base'] =  base_points
    return df_score,dict_bin_score,params_A,params_B,base_points


def score_params_cal(base_point, odds, PDO):
    ##给定预期分数，与翻倍分数，确定参数A,B
    B = PDO/np.log(2)
    A = base_point + B*np.log(odds)
    return A, B


def myfunc(x):
    return str(x[0])+'_'+str(x[1])


## 计算样本分数
def cal_score(df_1, dict_bin_score, dict_cont_bin, dict_disc_bin, base_points):
    ## 先对原始数据分箱映射，然后，用分数字典dict_bin_score映射分数，基础分加每项的分数就是最终得分
    df_1.reset_index(drop=True, inplace=True)
    df_all_score = pd.DataFrame()
    ## 连续变量
    for i in dict_cont_bin.keys():
        if i in dict_bin_score.keys():
            df_all_score = pd.concat(
                [df_all_score, varbin_meth.cont_var_bin_map(df_1[i], dict_cont_bin[i],i).map(dict_bin_score[i])], axis=1)
    ## 离散变量
    for i in dict_disc_bin.keys():
        if i in dict_bin_score.keys():
            df_all_score = pd.concat(
                [df_all_score, varbin_meth.disc_var_bin_map(df_1[i], dict_disc_bin[i],i).map(dict_bin_score[i])], axis=1)

    df_all_score.columns = [x.split(sep='_BIN')[0] for x in list(df_all_score.columns)]
    df_all_score['base_score'] = base_points
    df_all_score['score'] = df_all_score.apply(sum, axis=1)
    df_all_score['target'] = df_1.target
    return df_all_score

def qiza(tags):
    if pd.isnull(tags):
        return tags
    elif "a" in tags:
        tags = 1
    elif "b" in tags:
        tags = 2
    elif "c" in tags:
        tags = 3
    elif "d" in tags:
        tags = 4
    return tags




def communication(content):
    if pd.isnull(content):
        return content
    else:
        if content in "电信":
            content = content.replace("电信","0")
        elif content in "移动":
            content = content.replace("电信","1")
        elif content in "联通":
            content = content.replace("电信","2")

#**************************************读取数据**************************************************************************
#数据读取
if __name__ == '__main__':
    path = "E:/Python/BinHai_Rural_Commercial_bank/"
    # fileName = "hanhua_enterprise_eva.xlsx"
    fileName = "example.xlsx"
    data = pd.read_excel(path + fileName)
#
print("查看源数据行列信息",data.shape)
print("查看源数据好坏样本",data["target"].value_counts())


#
#数据探索性分析，缺失值统计
# eda_data = toad.detector.detect(data)
# eda_data.sort_values(by="missing",inplace=True,ascending=False)
# # eda_data.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/eda_data.xlsx")
print("toad eda探索性分析完成！")


data.rename(columns={'企业状态':'enterprise_status','成立日期':'establish_date','省份':'provice','注册资本':'registered_capital',
                      '企业类型':'enterprise_type','营业开始日期':'commencement_date','实缴资本':'contributed_capital'
                     },inplace = True) # 将 loan_status 名称改为 target（目标）

#1.删除与违约风险无关的变量
other_variable = ["内部KeyNo","法人名","登记机关","公司名称","返回码","进件名称_1","注册号","更新日期","法人名",
                  "社会统一信用代码","地址","经营范围","核准日期","公司logo","营业结束日期","序号","需求数据时点","客户名称",
                 "客户证件号","企业名称","客户标签","评分","吊销日期","组织机构代码","客户电话","pd_cell_province",
                  "pd_cell_city","pd_id_where","pd_id_city","provice","commencement_date","contributed_capital"]



data2 = data.drop(other_variable,axis=1)
print("删除变量后数据信息行列：",data2.shape)
print("删除后的列",data2.columns)

#2.删除缺失率大于等于90%的变量
data2 = missing_delete_var(data2,0.9)
print("删除缺失率大于90%的变量:",data2.shape)

#3.删除缺失率大于90%的样本企业，
data3 = missing_delete_user(data2,len(data2.columns)*0.9)
print("删除缺失率大于90%的样本:",data3.shape)


## 4.删除行全为缺失值的本样本
data3.dropna(axis=0, how='all', inplace=True)
print('删除行为全部缺失的数据：', data3.shape)


## 5.删除只有唯一值仅有一个的变量
cols_name = list(data3.columns)
cols_name.remove('target')
data4,dele_list = constant_del(data3, cols_name) # 调用定义好的子函数
print('删除仅有一个唯一值的变量后：', data4.shape)

## 5.删除长尾数据
cols_name_1 = list(data4.columns)
cols_name_1.remove('target')
data4,dele_list = tail_del(data4,cols_name_1,rate=0.8) # 调用定义好的子函数


# #4.缺失值填充
# ##统计分类变量缺失比率
numeric_vars = data4.select_dtypes(include=["int","int32","int64","float","float32","float64"]).columns
numeric_vars =list(numeric_vars)
catgory_vars = [col for col in data4.columns if col not in numeric_vars]

print("类别变量",catgory_vars)


numeric_vars = data4.select_dtypes(include=["int","int32","int64","float","float32","float64"]).columns
catgory_vars = [col for col in data4.columns if col not in numeric_vars]


# # 6.运营数据替换
data4["pd_cell_type"] = data4["pd_cell_type"].apply(lambda x:communication(x))
data4["pd_cell_type"] = data4["pd_cell_type"].apply(lambda x:communication(x))
data4["pd_cell_type"] = data4["pd_cell_type"].apply(lambda x:communication(x))
#


# 5.日期转化为年份
now_date = datetime.datetime.now()
now_date.strftime("%Y-%m-%d")
data4["establish_date"] = data4["establish_date"].apply(lambda x : date_caculate(now_date,x))
print("日期转化完成！")

# 6.注册资本优化

# data4["contributed_capital"] = data4["contributed_capital"].apply(lambda x : unit_conver(x))
data4["registered_capital"] = data4["registered_capital"].apply(lambda x : unit_conver(x))
print("注册资本格式优化完成！")

# 7.企业经营状态替换
data4["enterprise_status"] = data4["enterprise_status"].apply(lambda x:encoding_type(x))
print("企业经营状态替换数据优化完成")


#8.企业类型
data4["enterprise_type"] = data4["enterprise_type"].apply(lambda x : encoding_data(x))
print("企业类型数据优化完成")

# data4["frg_group_num"] = data4["frg_group_num"].apply(lambda x : qiza(x))
# print("企业类型数据优化完成")


#
#
# #4.缺失值填充
# ##统计分类变量缺失比率
# numeric_vars = data4.select_dtypes(include=["int","int32","int64","float","float32","float64"]).columns
# catgory_vars = [col for col in data4.columns if col not in numeric_vars]
# for i in catgory_vars:
#     if i in "establish_date":
#         list(numeric_vars).append(i)
#     elif i in "enterprise_type":
#         list(numeric_vars).append(i)
#     elif i in "frg_group_num":
#         list(numeric_vars).append(i)

data4,drop_data = toad.selection.select(data4,data4["target"],empty=0.7,iv=0.03,corr=1,return_drop=True)
print(data4.shape)
# 企业类型数据优化完成
#
# # # 缺失值处理方法：
# # # 1.缺失率小于5%的特征利用均值进行插补
# # # 2.缺失率在5%-15%的随机森林进行插补
# # # 3.缺失率大于15%-90%的特征特征构造新变量
# # # 4.缺失率大于90%的特征直接删除
#
#
# missing_df = missing_cal(data4[numeric_vars])
# less_percent5_col = missing_df[(missing_df.missing_pct < 0.15) & (missing_df.missing_pct > 0)].col
# between_percent15_90_col = missing_df[missing_df.missing_pct >= 0.15].col
#
# #9.数值型数据缺失数据填充
# data4[numeric_vars] = numeric_interpolation(data4,"median",list(less_percent5_col.values))
# print("均值插补完成")
# data4[numeric_vars] = numeric_interpolation(data4,"rf",list(between_percent15_90_col.values))
# print("随机森林插补完成")
# print("数值型数据缺失数据填充完成！！！")
#
#
#
#
#
# # 7.删除只有一种状态的变量
# cols_name = list(data4.columns)
# cols_name.remove('target')
# data4,dele_list = constant_del(data4, cols_name) # 调用定义好的子函数
#
#
data4.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/rf/test/profile_standard.xlsx")
print("宽表写入完成！")
# # #
#
#
#
# #**************************************变量分箱**************************************************************************
#
#
# #
# for s in numeric_vars:
#     print('变量'+s+'可能取值'+str(len(data4[s].unique())))
#     if len(data4[s].unique())<=10:
#         catgory_vars.append(s)
#         numeric_vars.remove(s)
#
#         # 同时将后加的数值变量转为字符串
#         # 先用bool数值将变量取值中空值标注成 FALSE（0）
#         index_1 = data4[s].isnull()
#         # 如果变量中有空值，则对非空值进行字符串转化
#         if sum(index_1) > 0:
#             data4.loc[~index_1,s] = data4.loc[~index_1,s].astype('str')
#         # 如果变量中没有空值，则直接进行字符串转化
#         else:
#             data4[s] = data4[s].astype('str')
#         # index_2 = data_test[s].isnull()
#         # if sum(index_2) > 0:
#         #     data_test.loc[~index_2,s] = data_test.loc[~index_2,s].astype('str')
#         # else:
#         #     data_test[s] = data_test[s].astype('str')
#
#
# ##划分训练集和测试集
#
# # train_data,test_data = train_test_split(data4,test_size=0.2,stratify=data4.target,random_state=25)
# # print('训练集中，好信用 / 坏信用，比值：',sum(train_data.target==0)/train_data.target.sum())
# # print('测试集中，好信用 / 坏信用，比值：',sum(test_data.target==0)/test_data.target.sum())
# #
#
# data4.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/rf/profile_standard_establish_err.xlsx")
#
#
#
# ## 连续变量分箱
# dict_cont_bin = {}
# for i in numeric_vars:
#     print("变量i进行分箱",i)
#     dict_cont_bin[i], gain_value_save, gain_rate_save = varbin_meth.cont_var_bin(data4[i], data4.target, i,
#                                                                                  method=2, mmin=4, mmax=12,
#                                                                                  bin_rate=0.01, stop_limit=0.05,
#                                                                               bin_min_num=20)
# print("连续变量分箱完成")
# ## 离散变量分箱
# dict_disc_bin = {}
# del_key = []
# for i in catgory_vars:
#     dict_disc_bin[i], gain_value_save, gain_rate_save, del_key_1 = varbin_meth.disc_var_bin(train_data[i],
#                                                                                             train_data.target,i, method=2,
#                                                                                             mmin=4,
#                                                                                             mmax=10, stop_limit=0.05,
#                                                                                             bin_min_num=20)
#     # if len(del_key_1) > 0:
#     #     del_key.extend(del_key_1)
#
# print("离散变量分箱完成")
#
# ## 删除分箱数只有1个的变量
# if len(del_key) > 0:
#     for j in del_key:
#         del dict_disc_bin[j]
#
# print("删除分箱数只有1个的变量完成")
#
#
# ## 训练数据分箱
# ## 连续变量分箱映射
# df_cont_bin_train = pd.DataFrame()
# for i in dict_cont_bin.keys():
#     df_cont_bin_train = pd.concat([df_cont_bin_train, varbin_meth.cont_var_bin_map(data4[i], dict_cont_bin[i],i)],
#                                   axis=1)
# print("训练集离散变量分箱映射完成", df_cont_bin_train.shape)
# df_cont_bin_train.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/bn_model_ 2/df_cont_bin_train.xlsx")
# print("训练集连续变量分箱映射写入完成！！！")
# #
# ## 离散变量分箱映射
# #    ss = data_train[list( dict_disc_bin.keys())]
# df_disc_bin_train = pd.DataFrame()
# for i in dict_disc_bin.keys():
#     df_disc_bin_train = pd.concat([df_disc_bin_train, varbin_meth.disc_var_bin_map(train_data[i], dict_disc_bin[i],i)],
#                                   axis=1)
# print("训练集连续变量分箱映射完成", df_cont_bin_train.shape)
# df_disc_bin_train.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/bn_model_ 2/df_disc_bin_train.xlsx")
# print("训练集离散变量分箱映射写入完成！！！")
#
#
# ## 测试数据分箱
# ## 连续变量分箱映射
# df_cont_bin_test = pd.DataFrame()
# for i in dict_cont_bin.keys():
#     df_cont_bin_test = pd.concat([df_cont_bin_test, varbin_meth.cont_var_bin_map(test_data[i], dict_cont_bin[i],i)],
#                                  axis=1)
# print("测试集连离散变量分箱映射完成", df_cont_bin_train.shape)
# df_cont_bin_test.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/bn_model_ 2/df_cont_bin_test.xlsx")
# print("测试集连续变量分箱映射写入完成！！！")
#
#
# ## 离散变量分箱映射
# #    ss = data_test[list( dict_disc_bin.keys())]
# df_disc_bin_test = pd.DataFrame()
# for i in dict_disc_bin.keys():
#     df_disc_bin_test = pd.concat([df_disc_bin_test, varbin_meth.disc_var_bin_map(test_data[i], dict_disc_bin[i],i)],
#                                  axis=1)
#
# print("测试集连续变量分箱映射完成", df_cont_bin_train.shape)
# df_disc_bin_test.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/bn_model_ 2/df_disc_bin_test.xlsx")
# print("测试集离散变量分箱映射写入完成！！！")
#
#
# # 组成分箱后的训练集与测试集
# df_disc_bin_train['target'] = train_data.target
# data_train_bin = pd.concat([df_cont_bin_train, df_disc_bin_train], axis=1)
# df_disc_bin_test['target'] = test_data.target
# data_test_bin = pd.concat([df_cont_bin_test, df_disc_bin_test], axis=1)
#
# data_train_bin.reset_index(inplace=True, drop=True)
# data_test_bin.reset_index(inplace=True, drop=True)
# data_train_bin.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/bn_model_ 2/data_train_bin_.xlsx")
# data_test_bin.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/bn_model_ 2/data_test_bin_.xlsx")
#
#
# var_all_bin = list(data_train_bin.columns)
# var_all_bin.remove('target')
#
# print('训练集变量分箱结果：')
# data_train_bin
#
#
#
# data_train_bin = pd.concat([df_cont_bin_train, df_cont_bin_test], axis=0)
#
#
#
#
# print("进入WOE编码")
# ## WOE编码
# ## 训练集WOE编码
# df_train_woe, dict_woe_map, dict_iv_values ,var_woe_name = var_encode.woe_encode(data_train_bin,
#                             path,var_all_bin, data_train_bin.target,'dict_woe_map', flag='train')
#
# print("训练集WOE编码完成！！！")
# ## 测试集WOE编码
# df_test_woe, var_woe_name = var_encode.woe_encode(data_test_bin,
#                             path,var_all_bin, data_test_bin.target, 'dict_woe_map',flag='test')
#
# print("测试集WOE编码完成！！！")
#
#
#
#
#
#
# ## IV值初步筛选，选择iv大于等于0.01的变量
# df_train_woe = iv_selection_func(df_train_woe, dict_iv_values, iv_low=0.01)
#
# df_train_woe.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/bn_model_ 2/df_train_woe.xlsx")
# print("IV值初步筛选，选择iv大于等于0.01的变量写入完成！！！")
#
# ## 相关性分析，相关系数即皮尔逊相关系数大于0.8的，删除IV值小的那个变量。
# sel_var = list(df_train_woe.columns)
# sel_var.remove('target')
# ## 循环，变量与多个变量相关系数大于0.8，则每次只删除IV值最小的那个，直到没有大于0.8的变量为止
# while True:
#     pearson_corr = (np.abs(df_train_woe[sel_var].corr()) >= 0.8)
#     if pearson_corr.sum().sum() <= len(sel_var):
#         break
#     del_var = []
#     for i in sel_var:
#         var_1 = list(pearson_corr.index[pearson_corr[i]].values)
#         if len(var_1) > 1:
#             df_temp = pd.DataFrame({'value': var_1, 'var_iv': [dict_iv_values[x.split(sep='_woe')[0]] for x in var_1]})
#             del_var.extend(list(df_temp.value.loc[df_temp.var_iv == df_temp.var_iv.min(),].values))
#     del_var1 = list(np.unique(del_var))
#     ## 删除这些，相关系数大于0.8的变量
#     sel_var = [s for s in sel_var if s not in del_var1]
# print('=' * 80)
# print('IV值筛选后，剩余变量个数', len(sel_var), '个；较筛选前少了', 95 - len(sel_var), '个')
#


#
# ## 随机森林排序
# ## 特征选择
# fs = FeatureSelector(data = df_train_woe[sel_var], labels = data_train_bin.target)
# ## 一次性去除所有的不满足特征
# fs.identify_all(selection_params = {'missing_threshold': 0.9,
#                                     'correlation_threshold': 0.8,
#                                     'task': 'classification',
#                                     'eval_metric': 'binary_error',
#                                     'max_depth':2,
#                                     'cumulative_importance': 0.90})
# df_train_woe = fs.remove(methods = 'all')
# df_train_woe['target'] = data_train_bin.target
#
# df_train_woe.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/bn_model_ 2/df_train_woe2.xlsx")
# print("随机森林排序后变量写入完成！！！")
#
#
# print('原始训练集中，好/坏样本比例为',int(sum(train_data.target==0)/train_data.target.sum()),': 1，需要做样本均衡处理')
#
# var_woe_name = list(df_train_woe.columns)
# var_woe_name.remove('target')
#
# print("样本不均衡数据训练")
# ## 随机抽取一些好样本，与坏样本合并，再用 SMOTE 生成一个新的样本训练集
# df_temp_normal = df_train_woe[df_train_woe.target == 0]  # 筛出好样本
# df_temp_normal.reset_index(drop=True, inplace=True)
# index_1 = np.random.randint(low=0, high=df_temp_normal.shape[0] - 1, size=20000)
# index_1 = np.unique(index_1)
# # print('在含有',df_train_woe.shape[0],'个样本的训练集中，随机抽取 20000 个 target = 0 的好样本，去重后剩余的随机好样本量为：',len(index_1))
#
# df_temp = df_temp_normal.loc[index_1]
# index_2 = [x for x in range(df_temp_normal.shape[0]) if x not in index_1]  # 剩余没有被随机抽到的好样本
# df_temp_other = df_temp_normal.loc[index_2]
# df_temp = pd.concat([df_temp, df_train_woe[df_train_woe.target == 1]], axis=0, ignore_index=True)
# # print('用剩余好样本 + 坏样本 = 新样本集，样本量为：',df_temp.shape[0])
#
# ## 用随机抽取的样本做样本生成
# sm_sample_1 = SMOTE(random_state=10, sampling_strategy=1, k_neighbors=5)  # kind='borderline1'，ratio=0.5
# x_train, y_train = sm_sample_1.fit_resample(df_temp[var_woe_name], df_temp.target)
# print('用 SMOTE 在随机样本集中，构建 K=5 的近邻领域，并在其中生成少数样本后，样本总量为：', x_train.shape[0])
#
# ## 合并数据
# x_train = np.vstack([x_train, np.array(df_temp_other[var_woe_name])])
# y_train = np.hstack([y_train, np.array(df_temp_other.target)])
#
# print('最终进行训练的样本集中，好/坏样本比例为', int(sum(y_train == 0) / sum(y_train)), ': 1')
#
# del_list = []
# for s in var_woe_name:
#     index_s = df_test_woe[s].isnull()
#     if sum(index_s) > 0:
#         del_list.extend(list(df_test_woe.index[index_s]))
# if len(del_list) > 0:
#     list_1 = [x for x in list(df_test_woe.index) if x not in del_list]
#     df_test_woe = df_test_woe.loc[list_1]
#
#     x_test = df_test_woe[var_woe_name]
#     x_test = np.array(x_test)
#     y_test = np.array(df_test_woe.target.loc[list_1])
# else:
#     x_test = df_test_woe[var_woe_name]
#     x_test = np.array(x_test)
#     y_test = np.array(df_test_woe.target)
#


### logistic 逻辑回归模型建模
## 设置待优化的超参数
# lr_param = {'C': [0.01, 0.1, 0.2, 0.5, 1, 1.5, 2],
#             'class_weight': [{1: 1, 0: 1},  {1: 2, 0: 1}, {1: 3, 0: 1}, {1: 5, 0: 1}]}
# ## 初始化网格搜索
# lr_gsearch = GridSearchCV(
#         estimator=LogisticRegression(random_state=0, fit_intercept=True, penalty='l2', solver='saga'),
#         param_grid=lr_param, cv=3, scoring='f1', n_jobs=-1, verbose=2)
# ## 执行超参数优化
# lr_gsearch.fit(x_train, y_train)
# print('LR逻辑回归模型最优得分 {0},\n最优参数{1}'.format(lr_gsearch.best_score_,lr_gsearch.best_params_))
#
# ## 用最优参数，初始化logistic模型
# LR_model = LogisticRegression(penalty='l2', solver='saga',
#                                 class_weight=lr_gsearch.best_params_['class_weight'],max_iter=10000)
# ## 训练logistic模型
# #    LR_model = LogisticRegression(C=0.01, penalty='l2', solver='saga',
# #                                    class_weight={1: 3, 0: 1})
#

# train_new2 = toad.selection.stepwise(train_new1,target="target",direction="both",criterion="AIC")
# print(train_new2.shape)
# print(train_new2.columns)
#
#
# x=train_new6.drop('target',axis=1) #设定自变量
# y=train_new6["target"] #设定因变量
#
# train_x,test_x,train_y,test_y=train_test_split(x,y,train_size=0.8,random_state=4)
#
#
LR_model_fit = LogisticRegression()
result = LR_model_fit.fit(train_x, train_y)

pred_y=LR_model_fit.predict(test_x)  #预测测试集的y
result.score(test_x,test_y)    #计算预测精度 正确率
#
#
#
# #利用sklearn.metrics计算ROC和AUC值
# from sklearn.metrics import  roc_curve, auc  #导入函数
# proba_y=LR_model_fit.predict_proba(test_x)  #预测概率predict_proba：
# '''返回的是一个n行k列的数组，第i行第j列上的数值是模型预测第i个预测样本的标签为j的概率，此时每一行的和应该等于1。'''
# fpr,tpr,threshold=roc_curve(test_y,proba_y[:,1])  #计算threshold阈值，tpr真正例率，fpr假正例率，大于阈值的视为1即坏客户
# roc_auc=auc(fpr,tpr)   #计算AUC值
# ks_test=abs(fpr - tpr).max()
#
#
# plt.plot(fpr,tpr,'b',label= 'AUC= %0.2f' % roc_auc) #生成roc曲线
# plt.legend(loc='lower right')
# plt.plot([0,1],[0,1],'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.ylabel('真正率')
# plt.xlabel('假正率')
# plt.show()
# print("AUC值",roc_auc)
# print("K-S值：",ks_test)



#
# ## 模型评估
# y_pred = LR_model_fit.predict(x_test)
# ## 计算混淆矩阵与recall、precision
# cnf_matrix = pd.DataFrame(data=confusion_matrix(y_test, y_pred),
#                           columns=['预测结果为正例','预测结果为反例'],index=['真实样本为正例','真实样本为反例'])
# recall_value = recall_score(y_test, y_pred)
# precision_value = precision_score(y_test, y_pred)
# acc = accuracy_score(y_test, y_pred)
# print('LR 逻辑回归模型的召回率：',recall_value)
# print('LR 逻辑回归模型的精准率：',precision_value)
# print('LR 逻辑回归模型预测的正确率:',acc)
# print('='*80)
# print('测试集的混淆矩阵：')
# cnf_matrix
#

#
#
#
#
#
# ## 保存模型的参数用于计算评分
# var_woe_name.append('intercept')
# ## 提取权重
# weight_value = list(LR_model_fit.coef_.flatten())
# ## 提取截距项
# weight_value.extend(list(LR_model_fit.intercept_))
# dict_params = dict(zip(var_woe_name,weight_value))
#
#
# ## 提取 训练集、测试集 样本分数
# y_score_train = LR_model_fit.predict_proba(x_train)[:, 1]
# y_score_test = LR_model_fit.predict_proba(x_test)[:, 1]
#
#
#
# ## 生成评分卡
# df_score,dict_bin_score,params_A,params_B,score_base = create_score(dict_woe_map,
#                                                 dict_params,dict_cont_bin,dict_disc_bin)
# print('参数 A 取值:',params_A)
# print('='*80)
# print('参数 B 取值:',params_B)
# print('='*80)
# print('基准分数:',score_base)
#
#
#
# var_bin_score = pd.DataFrame(dict_bin_score)
# print('全部',var_bin_score.shape[1],'个变量，不同取值 bins(分箱) 所对应的分数：')
# var_bin_score.sort_index()
#
#
# ## 计算样本评分
# df_all = pd.concat([train_data,test_data],axis = 0)
# df_all_score = cal_score(df_all,dict_bin_score,dict_cont_bin,dict_disc_bin,score_base)
# df_all_score.score[df_all_score.score >900] = 900
# print('样本最高分：',df_all_score.score.max())
# print('样本最低分：',df_all_score.score.min())
# print('样本平均分：',df_all_score.score.mean())
# print('样本中位数得分：',df_all_score.score.median())
# print('='*80)
# print('全部样本的变量得分情况：')
# df_all_score
#
#
# ## 评分卡区间分数统计
# good_total = sum(df_all_score.target == 0)
# bad_total = sum(df_all_score.target == 1)
# score_bin = np.arange(300,950,50)
# bin_rate = []
# bad_rate = []
# ks = []
# good_num = []
# bad_num = []
# for i in range(len(score_bin)-1):
#     ## 取出分数区间的样本
#     if score_bin[i+1] == 900:
#         index_1 = (df_all_score.score >= score_bin[i]) & (df_all_score.score <= score_bin[i+1])
#     else:
#         index_1 = (df_all_score.score >= score_bin[i]) & (df_all_score.score < score_bin[i+1])
#     df_temp = df_all_score.loc[index_1,['target','score']]
#     ## 计算该分数区间的指标
#     good_num.append(sum(df_temp.target==0))
#     bad_num.append(sum(df_temp.target==1))
#     ## 区间样本率
#     bin_rate.append(df_temp.shape[0]/df_all_score.shape[0]*100)
#     ## 坏样本率
#     bad_rate.append(df_temp.target.sum()/df_temp.shape[0]*100)
#     ## 以该分数为注入分数的ks值
#     ks.append(sum(bad_num[0:i+1])/bad_total - sum(good_num[0:i+1])/good_total )
#
#
# index_range = ['[ 300-350 )','[ 350-400 )','[ 400-450 )','[ 450-500 )','[ 500-550 )','[ 550-600 ) ','[ 600-650 ) ',
#                '[ 650-700 )','[ 700-750 )','[ 750-800 )','[ 800-850 )','[ 850-900 ]']
# df_result = pd.DataFrame({'好信用数量':good_num,'坏信用数量':bad_num,'区间样本率':bin_rate,
#                             '坏信用率':bad_rate,'KS值(真正率-假正率)':ks},index=index_range)
# print('评分卡12个区间分数统计结果如下：')
# df_result
#
#
# score_all = df_all_score.loc[:,['score','target']]
# target_0_score = score_all.loc[score_all['target']==0]
# target_1_score = score_all.loc[score_all['target']==1]
#
#
#
# print('观察不同分数段中，好坏信用样本频数分布:\n好样本采用左侧纵轴刻度，坏样本采用右侧纵轴刻度')
#
# bar_width = 0.3
# fig, ax1 = plt.subplots(figsize=(15,6))
# plt.bar(np.arange(0,12)+ bar_width,df_result.iloc[:,0],bar_width,alpha=0.7,color='blue', label='好信用样本量')
# ax1.set_ylabel('好信用样本频数',fontsize=17)
# ax1.set_ylim([0,60000])
# plt.grid(True)
# plt.xlabel('分值区间',fontsize=17)
# plt.xticks(np.arange(0,12),index_range,rotation=35,fontsize=15)
# plt.yticks(fontsize=17)
#
# # 共享横轴，双纵轴
# ax2 = ax1.twinx()
# ax2.bar(np.arange(0,12),df_result.iloc[:,1],bar_width,alpha=0.7,color='red', label='坏信用样本量')
# ax2.set_ylabel('坏信用样本频数',fontsize=17)
# ax2.set_ylim([0,500])
# plt.yticks(fontsize=17)
# plt.xlabel('分值区间',fontsize=17)
#
# # 合并图例
# handles1, labels1 = ax1.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# plt.legend(handles1+handles2, labels1+labels2, loc='upper right',fontsize=17)
# plt.show()
#
#
#
# print('观察不同分数段中，好坏信用样本的概率密度分布：')
# plt.figure(figsize=(15,6))
# plt.hist(target_0_score.iloc[:,0],bins=200,alpha=0.5,label='好信用',
#          color='blue',range=(300,900),density=True,rwidth=0.3,histtype='stepfilled')
# plt.hist(target_1_score.iloc[:,0],bins=200,alpha=0.5,label='坏信用',
#          color='red',range=(300,900),density=True,rwidth=0.3,histtype='stepfilled')
# plt.xticks(fontsize=17)
# plt.yticks(fontsize=17)
# plt.xlabel('分值区间',fontsize=17)
# plt.ylabel('概率密度',fontsize=17)
# plt.legend(fontsize=17)
# plt.show()
#
#
