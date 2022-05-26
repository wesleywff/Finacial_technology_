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


def data_read(data_path,file_name):
    # df = pd.read_csv( os.path.join(data_path, file_name), delim_whitespace = True, header = None )
    df = pd.read_excel(data_path + file_name)

    # columns = ['status_account','duration','credit_history','purpose', 'amount',
    #            'svaing_account', 'present_emp', 'income_rate', 'personal_status',
    #            'other_debtors', 'residence_info', 'property', 'age',
    #            'inst_plans', 'housing', 'num_credits',
    #            'job', 'dependents', 'telephone', 'foreign_worker', 'target']
    # df.columns = columns
    # 将标签变量由状态1,2转为0,1; 其中0表示好信用，1表示坏信用
    df.target = df["target"]
    # 将数据分为data_train（训练集）和 data_test（测试集）两部分
    # 按目标变量进行分层抽样，即训练集和测试集中，好坏样本的比例相同。
    # data_train, data_test = train_test_split(df, test_size=0.2, random_state=0,stratify=df.target)
    # 由于训练集、测试集是随机划分，索引是乱的，需要重新排序
    # data_train = data_train.reset_index(drop=True)
    # data_test = data_test.reset_index(drop=True)
    return df

if __name__ == '__main__':
    # data_path = "E:/Python/BinHai_Rural_Commercial_bank/bn_rural_commercal_bank_model/"
    data_path = "E:/Python/BinHai_Rural_Commercial_bank/rf/"

    # E:/Python/BinHai_Rural_Commercial_bank/
    file_name = 'data_s.xlsx'
    # 读取数据
    data4 = data_read(data_path,file_name)


data4.info()


print('训练集中"好信用target=0"样本量：',sum(data4.target ==0))
print('训练集中"坏信用target=1"样本量：',sum(data4.target ==1))



data5,drop_list = toad.selection.select(data4,data4["target"],empty=0.7,iv=0.2,corr=0.7,return_drop=True)
print("数据初筛后样本特征",data5.shape)

data_train, data_test = train_test_split(data5, test_size=0.2, random_state=0,stratify=data5.target)
# 由于训练集、测试集是随机划分，索引是乱的，需要重新排序
data_train = data_train.reset_index(drop=True)
data_test = data_test.reset_index(drop=True)





# 定义函数（1）：" 离散变量 / 连续变量 " 区分函数
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


# 实现：" 离散变量 vs 连续变量 " 分离
#
# a. 对于取值小于10种数值的连续型变量，改为离散变量，且将数值转化为字符串；
#
# b. 在对上述连续变量进行离散变量转化的过程中，先要将变量中的空值进行bool数标注，只将取值中非空的数值转换为字符串，保留空值属性；
#
# c. 注意：有些变量从表象上看是数值型连续变量，但从业务角度任务是离散型变量，如借款期限12/18/24/36期，这时需要主动将这些数值型变量改为字符串型离散变量。


feature_names = list(data5.columns)
feature_names.remove('target')
#
categorical_var,numerical_var = category_continue_separation(data5,feature_names)
print('初始7个连续变量：','\n',numerical_var)
print('初始7个连续变量：','\n',categorical_var)
# numerical_var =["ae_m3_id_nbank_cons_allnum","ae_m3_id_nbank_min_monnum"]
# categorical_var=["pef_paypower_prov","ae_m3_cell_bank_weekend_allnum","als_m1_cell_bank_allnum","als_m3_id_bank_min_monnum"]
# print('='*80)
print('初始连续变量情况：')
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

print('='*80)
print('经转换后，剩余连续变量：','\n',numerical_var)
print('='*80)
print('离散变量：','\n',categorical_var)
# data_train.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/rf/test.xlsx")

#
# ## 连续变量不同类别下的分布
# j=1
# plt.figure(figsize=(16,24))
# for i in numerical_var:
#     ##取非缺失值的数据（先用布尔值进行索引判断，再对具体列取值）
#     df_temp = data_train.loc[~data_train[i].isnull(),[i,'target']]
#     df_good = df_temp[df_temp.target == 0]
#     df_bad = df_temp[df_temp.target == 1]
#     ## 计算统计量
#     valid = round(df_temp.shape[0]/data_train.shape[0]*100,2) # 非空数据占比，即有效数据占比
#     Mean = round(df_temp[i].mean(),2)
#     Std = round(df_temp[i].std(),2)
#     Max = round(df_temp[i].max(),2)
#     Min = round(df_temp[i].min(),2)
#     ## 统计性描述绘图
#     plt.subplot(4,2,j)
#     plt.hist(df_good[i],bins =20, alpha=0.5,label='好样本')
#     plt.hist(df_bad[i],bins =20, alpha=0.5,label='坏样本')
#     plt.ylabel(i,fontsize=16)
#     plt.title( 'valid rate='+str(valid)+'%, Mean='+str(Mean) + ', Std='+str(Std)+', Max='+str(Max)+', Min='+str(Min),fontsize=14)
#     plt.legend(fontsize=15)
#     j=j+1
#
#
# # 变量分箱
#
#
# ##离散变量不同类别下的分布
# k=1
# plt.figure(figsize=(18,30))
# for i in categorical_var:
#     ##非缺失值数据
#     df_temp = data_train.loc[~data_train[i].isnull(),[i,'target']]
#     df_bad = df_temp[df_temp.target == 1]
#     valid = round(df_temp.shape[0]/data_train.shape[0]*100,2)
#     bad_rate = []
#     bin_rate = []
#     var_name = []
#     for j in data_train[i].unique():
#         if pd.isnull(j):
#             df_1 = data_train[data_train[i].isnull()]
#             bad_rate.append(sum(df_1.target)/df_1.shape[0])
#             bin_rate.append(df_1.shape[0]/data_train.shape[0])
#             var_name.append('NA')
#         else:
#             df_1 = data_train[data_train[i] == j]
#             bad_rate.append(sum(df_1.target)/df_1.shape[0])
#             bin_rate.append(df_1.shape[0]/data_train.shape[0])
#             var_name.append(j)
#     df_2 = pd.DataFrame({'var_name':var_name,'bin_rate':bin_rate,'bad_rate':bad_rate})
#     ## 绘图
#     plt.subplot(6,3,k)
#     plt.bar(np.arange(1,df_2.shape[0]+1),df_2.bin_rate,0.1,color='black',alpha=0.5, label='类别占样本量比')
#     plt.xticks(np.arange(1,df_2.shape[0]+1), df_2.var_name)
#     plt.plot( np.arange(1,df_2.shape[0]+1),df_2.bad_rate,  color='green', alpha=0.5,label='坏样本比率')
#     plt.ylabel(i,fontsize=16)
#     plt.title( 'valid rate='+str(valid)+'%',fontsize=14)
#     plt.legend(fontsize=15)
#     k=k+1

# 连续变量分箱
dict_cont_bin = {}
for i in numerical_var:
    dict_cont_bin[i], gain_value_save, gain_rate_save = varbin_meth.cont_var_bin(data_train[i],
                                                                                 data_train.target,i, method=2, mmin=3,
                                                                                 mmax=12,
                                                                                 bin_rate=0.01, stop_limit=0.05,
                                                                                 bin_min_num=20)

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

print('删除分箱数只有单个的变量：', '共', len(del_key), '个')
print(del_key)


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



#
### 测试集数据分箱
# 连续变量分箱映射
if bool(dict_cont_bin):
    df_cont_bin_test = pd.DataFrame()
    for i in dict_cont_bin.keys():
        df_cont_bin_test = pd.concat([df_cont_bin_test,
                                      varbin_meth.cont_var_bin_map(data_test[i],
                                                                   dict_cont_bin[i], i)], axis=1)


# 离散变量分箱映射
#    ss = data_test[list( dict_disc_bin.keys())]
if bool(dict_disc_bin):
    df_disc_bin_test = pd.DataFrame()
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
print('训练集18个变量分箱结果：')
data_train_bin




# **********************************************************************************************************************
# WOE编码
var_all_bin = list(data_train_bin.columns)
var_all_bin.remove('target')

# WOE编码
var_all_bin = list(data_train_bin.columns)
var_all_bin.remove("target")

# 训练集WOE编码
df_train_woe, dict_woe_map, dict_iv_values ,var_woe_name = var_encode.woe_encode(data_train_bin,data_path,
                                var_all_bin, data_train_bin.target,'dict_woe_map', flag='train')

# 测试集WOE编码
df_test_woe, var_woe_name = var_encode.woe_encode(data_test_bin,data_path,var_all_bin,
                                                  data_test_bin.target, 'dict_woe_map',flag='test')

print('训练集WOE编码:')



print('查看训练集数据，不同变量的IV值：（共18个）')





#
# # 设置优化参数
# # C 为正则项惩罚系数（7组）
# # class_weight 用字典形式为样本加权，来抑制样本不均衡（3组）
# lr_param = {'C': [0.01, 0.1, 0.2, 0.5, 1, 1.5, 2],
#             'class_weight': [{1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}]}
#
# # 初始化网格搜索
# # 选择 L2 正则项来抑制过拟合，选择 saga 作为优化算法
# # cv=3 进行三折交叉验证，由于有7组超参数 C，3组超参数 class_weight，因此共需3*7*3=63次拟合
# lr_gsearch = GridSearchCV(
#   estimator=LogisticRegression(random_state=0, fit_intercept=True, penalty='l2', solver='saga'),
#   param_grid=lr_param, cv=3, scoring='f1', n_jobs=-1, verbose=2)
#
# #执行超参数优化
# lr_gsearch.fit(x_train, y_train)
# print('logistic model best_score_ is {0},and best_params_ is {1}'.format(lr_gsearch.best_score_,
#                                                                             lr_gsearch.best_params_))
#
#
#


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


df_score,dict_bin_score,params_A,params_B,score_base = create_score(dict_woe_map,dict_params,dict_cont_bin,dict_disc_bin)
print('参数 A 取值:',params_A)
print('='*80)
print('参数 B 取值:',params_B)
print('='*80)
print('基准分数:',score_base)
df_score.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/bn_rural_commercal_bank_model2/df_all_score.xlsx")



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

print(df_score.head(5))




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

