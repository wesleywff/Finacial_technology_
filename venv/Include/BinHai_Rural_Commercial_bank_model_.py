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
from sklearn.metrics import roc_curve, auc,confusion_matrix,recall_score,precision_score,accuracy_score,r2_score
from toad.metrics import KS, F1, AUC





#**************************************一、函数**************************************************************************

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

# 生成评分卡函数
def create_score(dict_woe_map,dict_params,dict_cont_bin,dict_disc_bin):
    # 假设 Odds 在 1:60 时对应的参考分值为 600 分，分值调整刻度 PDO 为 20，则计算得到分值转化的参数 B = 28.85，A= 481.86。
    params_A,params_B = score_params_cal(base_point=600, odds=1/60, PDO=30)
    # 计算基础分
    base_points = round(params_A - params_B * dict_params['intercept'])
    df_score = pd.DataFrame()
    dict_bin_score = {}
    for k in dict_params.keys():
        if k !='intercept':
            df_temp =  pd.DataFrame([dict_woe_map[k.split(sep='_woe')[0]]]).T
            df_temp.reset_index(inplace=True)
            df_temp.columns = ['bin','woe_val']
            # 计算分值
            df_temp['score'] = round(-params_B*df_temp.woe_val*dict_params[k])
            dict_bin_score[k.split(sep='_BIN')[0]] = dict(zip(df_temp['bin'],df_temp['score']))
            # 连续变量的计算
            if k.split(sep='_BIN')[0] in dict_cont_bin.keys():
                df_1 = dict_cont_bin[k.split(sep='_BIN')[0]]
                df_1['var_name'] = df_1[['bin_low', 'bin_up']].apply(myfunc,axis=1)
                df_1 = df_1[['total', 'var_name','good','bad']]
                df_temp = pd.merge(df_temp , df_1,on='bin')
                df_temp['var_name_raw'] = k.split(sep='_BIN')[0]
                df_score = pd.concat([df_score,df_temp],axis=0)
            # 离散变量的计算
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

def category_continue_separation(df,feature_names):
    categorical_var = []
    numerical_var = []
    if 'target' in feature_names:
        feature_names.remove('target')
    # 先判断类型，如果是int或float就直接作为连续变量
    numerical_var = list(df[feature_names].select_dtypes(include=['int','float',
                                                                  'int32','float32',
                                                                  'int64','float64']).columns.values)
    categorical_var = [x for x in feature_names if x not in numerical_var]
    return categorical_var,numerical_var

# 公式转换函数
def score_params_cal(base_point, odds, PDO):
    # 给定基准分数 base_point、翻倍分数 PDO，确定参数 A、B
    B = PDO/np.log(2)
    A = base_point + B*np.log(odds)
    return A, B
def myfunc(x):
    return str(x[0])+'_'+str(x[1])



# 生成评分卡函数
def create_score(dict_woe_map,dict_params,dict_cont_bin,dict_disc_bin):
    # 假设 Odds 在 1:60 时对应的参考分值为 600 分，分值调整刻度 PDO 为 20，则计算得到分值转化的参数 B = 28.85，A= 481.86。
    params_A,params_B = score_params_cal(base_point=600, odds=1/60, PDO=30)
    # 计算基础分
    base_points = round(params_A - params_B * dict_params['intercept'])
    df_score = pd.DataFrame()
    dict_bin_score = {}
    for k in dict_params.keys():
        if k !='intercept':
            df_temp =  pd.DataFrame([dict_woe_map[k.split(sep='_woe')[0]]]).T
            df_temp.reset_index(inplace=True)
            df_temp.columns = ['bin','woe_val']
            # 计算分值
            df_temp['score'] = round(-params_B*df_temp.woe_val*dict_params[k])
            dict_bin_score[k.split(sep='_BIN')[0]] = dict(zip(df_temp['bin'],df_temp['score']))
            # 连续变量的计算
            if k.split(sep='_BIN')[0] in dict_cont_bin.keys():
                df_1 = dict_cont_bin[k.split(sep='_BIN')[0]]
                df_1['var_name'] = df_1[['bin_low', 'bin_up']].apply(myfunc,axis=1)
                df_1 = df_1[['total', 'var_name','good','bad']]
                df_temp = pd.merge(df_temp , df_1,on='bin')
                df_temp['var_name_raw'] = k.split(sep='_BIN')[0]
                df_score = pd.concat([df_score,df_temp],axis=0)
            # 离散变量的计算
            elif k.split(sep='_BIN')[0] in dict_disc_bin.keys():
                df_temp = pd.merge(df_temp , dict_disc_bin[k.split(sep='_BIN')[0]],on='bin')
                df_temp['var_name_raw'] = k.split(sep='_BIN')[0]
                df_score = pd.concat([df_score,df_temp],axis=0)

    df_score['score_base'] =  base_points
    return df_score,dict_bin_score,params_A,params_B,base_points



# 计算样本分数函数
def cal_score(df_1, dict_bin_score, dict_cont_bin, dict_disc_bin, base_points):
    # 先对原始数据分箱映射，然后，用分数字典dict_bin_score映射分数，基础分加每项的分数就是最终得分
    df_1.reset_index(drop=True, inplace=True)
    df_all_score = pd.DataFrame()
    # 连续变量
    for i in dict_cont_bin.keys():
        if i in dict_bin_score.keys():
            df_all_score = pd.concat([df_all_score, varbin_meth.cont_var_bin_map(df_1[i],
                                                                                 dict_cont_bin[i],i).map(
                dict_bin_score[i])], axis=1)
    # 离散变量
    for i in dict_disc_bin.keys():
        if i in dict_bin_score.keys():
            df_all_score = pd.concat([df_all_score, varbin_meth.disc_var_bin_map(df_1[i],
                                                                                 dict_disc_bin[i],i).map(
                dict_bin_score[i])], axis=1)

    df_all_score.columns = [x.split(sep='_BIN')[0] for x in list(df_all_score.columns)]
    df_all_score['base_score'] = base_points
    df_all_score['score'] = df_all_score.apply(sum, axis=1)
    df_all_score['target'] = df_1.target
    df_all_score['score'] = df_all_score['score'].apply(lambda x:score_limited(x))
    return df_all_score

def score_limited(score):
    if score >= 900:
        score = 900
    elif score <= 300:
        score = 300
    return round(score,0)



# ***********************************************二、读取数据*************************************************************************

if __name__ == '__main__':

    data_path = "E:/Python/BinHai_Rural_Commercial_bank/bn_rural_commercal_bank_model2/"
    fileName = "hanhua_enterprise_eva.xlsx"
    # fileName = "example.xlsx"
    data = pd.read_excel(data_path + fileName)
#
print("查看源数据行列信息",data.shape)
print("查看源数据好坏样本",data["target"].value_counts())

#数据探索性分析，缺失值统计
# eda_data = toad.detector.detect(data)
# eda_data.sort_values(by="missing",inplace=True,ascending=False)
# # eda_data.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/bn_rural_commercal_bank_model/eda_data.xlsx")



# ***********************************************二、数据处理*************************************************************************


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

##删除长尾数据、集中度高的数据
# cols_name_1 = list(data4.columns)
# cols_name_1.remove('target')
# data4,dele_list = tail_del(data4,cols_name_1,rate=0.8) # 调用定义好的子函数


# # 6.运营数据替换
data4["pd_cell_type"] = data4["pd_cell_type"].apply(lambda x:communication(x))
data4["pd_cell_type"] = data4["pd_cell_type"].apply(lambda x:communication(x))
data4["pd_cell_type"] = data4["pd_cell_type"].apply(lambda x:communication(x))
print("运营数据更新完成！")


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



# data4.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/bn_rural_commercal_bank_model/profile_standard.xlsx")
# print("宽表写入完成！")

# ***********************************************三、数据集划分*************************************************************************
data5,drop_list = toad.selection.select(data4,data4["target"],empty=0.7,iv=0.2,corr=0.7,return_drop=True)
print("数据初筛后样本特征",data5.shape)

data_train, data_test = train_test_split(data5, test_size=0.3, random_state=0,stratify=data5.target)
# 由于训练集、测试集是随机划分，索引是乱的，需要重新排序
data_train = data_train.reset_index(drop=True)
data_test = data_test.reset_index(drop=True)

feature_names = list(data5.columns)
feature_names.remove('target')
categorical_var,numerical_var = category_continue_separation(data5,feature_names)
print('离散变量：','\n',numerical_var)
print('连续变量：','\n',categorical_var)
# numeric_var


print('='*80)
print('初始连续变量情况：')
# 对于取值小于10种数值的连续型变量，改为离散变量，且将数值转化为字符串
# 对于取值小于10种数值的连续型变量，改为离散变量，且将数值转化为字符串
for s in set(numerical_var):
    print('变量'+s+'可能取值'+str(len(data_train[s].unique())))
    if len(data_train[s].unique())<=10:
        categorical_var.append(s)
        numerical_var.remove(s)
        # 同时将后加的数值变量转为字符串
        # 先用bool数值将变量取值中空值标注成 FALSE（0）
        index_1 = data_train[s].isnull()
        # 如果变量中有空值，则对非空值进行字符串转化
        if sum(index_1) > 0:
            data_train.loc[~index_1,s] = data_train.loc[~index_1,s].astype('str')
        # 如果变量中没有空值，则直接进行字符串转化
        else:
            data_train[s] = data_train[s].astype('str')
        index_2 = data_test[s].isnull()
        if sum(index_2) > 0:
            data_test.loc[~index_2,s] = data_test.loc[~index_2,s].astype('str')
        else:
            data_test[s] = data_test[s].astype('str')


# ***********************************************三、特征分箱*************************************************************************

# 离散变量分箱
dict_disc_bin = {}
del_key = []
for i in categorical_var:
    dict_disc_bin[i], gain_value_save, gain_rate_save, del_key_1 = varbin_meth.disc_var_bin(data_train[i],
                                                                                            data_train.target,i, method=2,
                                                                                            mmin=3,
                                                                                            mmax=8, stop_limit=0.05,
                                                                                            bin_min_num=20)
    if dict_disc_bin[i].empty:
        dict_disc_bin.pop(i)
        break
    elif len(del_key_1) > 0:
        del_key.extend(del_key_1)

if len(del_key) > 0:
    for j in del_key:
        del dict_disc_bin[j]
print("离散变量分箱后数量",len(dict_disc_bin))

print('删除分箱数只有单个的变量：', '共', len(del_key), '个')
print(del_key)


# 连续变量分箱
dict_cont_bin = {}
for i in numerical_var:
    dict_cont_bin[i], gain_value_save, gain_rate_save = varbin_meth.cont_var_bin(data_train[i],
                                                                                 data_train.target,i, method=2, mmin=3,
                                                                                 mmax=12,
                                                                                 bin_rate=0.01, stop_limit=0.05,
                                                                                 bin_min_num=20)
print("连续变量分箱后数量",len(dict_cont_bin))





### 训练集数据分箱

# 连续变量分箱映射
df_cont_bin_train = pd.DataFrame()
if bool(dict_cont_bin):
    for i in dict_cont_bin.keys():
        df_cont_bin_train = pd.concat([df_cont_bin_train,
                                       varbin_meth.cont_var_bin_map(data_train[i],
                                                                    dict_cont_bin[i], i)], axis=1)

# 离散变量分箱映射
#    ss = data_train[list( dict_disc_bin.keys())]
df_disc_bin_train = pd.DataFrame()
if bool(dict_disc_bin):
    for i in dict_disc_bin.keys():
        df_disc_bin_train = pd.concat([df_disc_bin_train,
                                       varbin_meth.disc_var_bin_map(data_train[i],
                                                                    dict_disc_bin[i], i)], axis=1)




### 测试集数据分箱
# 连续变量分箱映射
df_cont_bin_test = pd.DataFrame()
if bool(dict_cont_bin):
    for i in dict_cont_bin.keys():
        df_cont_bin_test = pd.concat([df_cont_bin_test,
                                      varbin_meth.cont_var_bin_map(data_test[i],
                                                                   dict_cont_bin[i], i)], axis=1)


# 离散变量分箱映射
#    ss = data_test[list( dict_disc_bin.keys())]
df_disc_bin_test = pd.DataFrame()
if bool(dict_disc_bin):
    for i in dict_disc_bin.keys():
        df_disc_bin_test = pd.concat([df_disc_bin_test,
                                      varbin_meth.disc_var_bin_map(data_test[i],
                                                                   dict_disc_bin[i], i)], axis=1)

### 组成分箱后的训练集与测试集
df_disc_bin_train['target'] = data_train.target
data_train_bin = pd.concat([df_cont_bin_train, df_disc_bin_train], axis=1)
df_disc_bin_test['target'] = data_test.target
data_test_bin = pd.concat([df_cont_bin_test, df_disc_bin_test], axis=1)

print('=' * 80)
print('训练集变量分箱结果：')
data_train_bin




# **********************************************************************************************************************

# WOE编码
var_all_bin = list(data_train_bin.columns)
var_all_bin.remove('target')

# 训练集WOE编码
df_train_woe, dict_woe_map, dict_iv_values ,var_woe_name = var_encode.woe_encode(data_train_bin,data_path,
                                var_all_bin, data_train_bin.target,'dict_woe_map', flag='train')

# 测试集WOE编码
df_test_woe, var_woe_name = var_encode.woe_encode(data_test_bin,data_path,var_all_bin,
                                                  data_test_bin.target, 'dict_woe_map',flag='test')



#*************************************************************************************************************************




df_train_woe = toad.selection.stepwise(df_train_woe,target="target",direction="both",criterion="AIC")
df_test_woe = df_test_woe[df_train_woe.columns]


x_train = df_train_woe.loc[:,df_train_woe.columns!="target"]
y_train = df_train_woe.loc[:,"target"]

x_test = df_test_woe.loc[:,df_train_woe.columns!="target"]
y_test = df_test_woe.loc[:,"target"]


#模型训练
LR_model_fit = LogisticRegression()
LR_model_fit.fit(x_train, y_train)


#1）训练集预测
EYtrain_proba = LR_model_fit.predict_proba(x_train)[:,1]
EYtrain = LR_model_fit.predict(x_train)
#2）常用的评估指标有F1、KS、AUC
fpr, tpr, thresholds = roc_curve(y_train,EYtrain_proba)

## 计算AR、gini等
roc_auc = auc(fpr, tpr)
ks = max(tpr - fpr)
ar = 2*roc_auc-1



print('F1:', F1(EYtrain_proba,y_train))
print('KS:', KS(EYtrain_proba,y_train))
print('AUC:', AUC(EYtrain_proba,y_train))
print('gini:', ar)
print("R-Squre",r2_score(y_train,EYtrain_proba))





print("测试数据：")
EYtest_proba = LR_model_fit.predict_proba(x_test)[:,1]
EYtest = LR_model_fit.predict(x_test)
## 计算fpr与tpr
fpr, tpr, thresholds = roc_curve(y_test, EYtest_proba)
## 计算AR、gini等
roc_auc = auc(fpr, tpr)
ks = max(tpr - fpr)
ar = 2*roc_auc-1


print('gini:',ar)
print('F1:', F1(EYtest_proba,y_test))
print('KS:', KS(EYtest_proba,y_test))
print('AUC:', AUC(EYtest_proba,y_test))
print("R-Squre",r2_score(y_test,EYtest_proba))



#保存模型的参数用于计算评分
final_name = list(df_train_woe.columns)
final_name.remove("target")
final_name.append('intercept')

#提取权重
weight_value = list(LR_model_fit.coef_.flatten())
weight_value.extend(list(LR_model_fit.intercept_))


dict_params = dict(zip(final_name,weight_value))

# 字典转换为DataFrame
LR_model_params=pd.DataFrame.from_dict(dict_params,orient='index',columns=['变量权重'])
print('LR逻辑回归模型生成的参数如下：')
LR_model_params.sort_values(by="变量权重",ascending=False)
LR_model_params.sort_values(by="变量权重",ascending=False).to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/bn_rural_commercal_bank_model2/LR_model_params.xlsx")



df_score,dict_bin_score,params_A,params_B,score_base = create_score(dict_woe_map,dict_params,dict_cont_bin,dict_disc_bin)
print('参数 A 取值:',params_A)
print('='*80)
print('参数 B 取值:',params_B)
print('='*80)
print('基准分数:',score_base)

df_score.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/bn_rural_commercal_bank_model2/df_score.xlsx")
# 计算总体样本评分
df_all = pd.concat([data_train,data_test],axis = 0)
df_all_score = cal_score(df_all,dict_bin_score,dict_cont_bin,dict_disc_bin,score_base)
print('样本最高分：',df_all_score.score.max())
print('样本最低分：',df_all_score.score.min())
print('样本平均分：',df_all_score.score.mean())
print('样本中位数得分：',df_all_score.score.median())
print('='*80)
print('全部样本的变量得分情况：')



df_all_score.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/bn_rural_commercal_bank_model2/df_all_score.xlsx")




#分数区间计算
score_bin = np.arange(300,950,50)
good_total = sum(df_all_score.target == 0)
bad_total = sum(df_all_score.target == 1)
bin_rate = []
bad_rate = []
ks = []
good_num = []
bad_num = []
for i in range(len(score_bin)-1):
    #取出分数区间的样本
    if score_bin[i+1] == 900:
        index_1 = (df_all_score.score >= score_bin[i]) & (df_all_score.score <= score_bin[i+1])
    else:
        index_1 = (df_all_score.score >= score_bin[i]) & (df_all_score.score < score_bin[i+1])
    df_temp = df_all_score.loc[index_1,['target','score']]
    #计算该分数区间的指标
    good_num.append(sum(df_temp.target==0))
    bad_num.append(sum(df_temp.target==1))
    #区间样本率
    bin_rate.append(df_temp.shape[0]/df_all_score.shape[0]*100)
    #坏样本率
    bad_rate.append(df_temp.target.sum()/df_temp.shape[0]*100)
    #以该分数为注入分数的ks值
    ks.append(sum(bad_num[0:i+1])/bad_total - sum(good_num[0:i+1])/good_total )

df_result = pd.DataFrame({'好信用数量':good_num,'坏信用数量':bad_num,'区间样本率':bin_rate,
                            '坏信用率':bad_rate,'KS值(真正率-假正率)':ks},index=zip((np.arange(300,900,50)),(np.arange(300,900,50))))
print('评分卡10个区间分数统计结果如下：')
df_result
df_result.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/bn_rural_commercal_bank_model2/data_train_distribute.xlsx")

