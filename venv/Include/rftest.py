
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

boston = load_boston()
x_full = boston.data  # x.shape = (506, 13)，共有506个样本，每个样本有13个特征
y_full = boston.target
n_samples = x_full.shape[0]
n_features = x_full.shape[1]

rng = np.random.RandomState(0)
missing_rate = 0.5
n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))
missing_features = rng.randint(0, n_features, n_missing_samples)  # 列索引 13
missing_samples = rng.randint(0, n_samples, n_missing_samples)  # 行索引 506

# 创造缺失的数据集
x_missing = x_full.copy()
y_missing = y_full.copy()
x_missing[missing_samples, missing_features] = np.nan
x_missing = pd.DataFrame(x_missing)



# 数据集中缺失值从少到多进行排序
x_missing_reg = x_missing.copy()
x_missing_reg.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/rf/x_missing_reg.xlsx")
print("x_missing_reg写入完成！")

sortindex = np.argsort(x_missing_reg.isnull().sum(axis=0)).values
f = 0
for i in sortindex:
    # 构建新特征矩阵和新标签
    df = x_missing_reg
    fillc = df.iloc[:, i]
    fillc.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/rf/fillc.xlsx")
    print("fillc写入完成！")

    df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_full)], axis=1)

    # 在新特征矩阵中，对含有缺失值的列，进行0的填补
    df_0 = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0).fit_transform(df)
    df_1 = pd.DataFrame(df_0)
    df_1.to_excel(r"E:/Python/BinHai_Rural_Commercial_bank/rf/df_1.xlsx")
    print("df_1写入完成！")


    # 构建新的训练集和测试集
    y_train = fillc[fillc.notnull()]
    y_test = fillc[fillc.isnull()]
    x_train = df_0[y_train.index, :]
    x_test = df_0[y_test.index, :]

    # 用随机森林填补缺失值
    rfc = RandomForestRegressor()
    rfc.fit(x_train, y_train)
    y_predict = rfc.predict(x_test)  # 用predict接口将x_test导入，得到我们的预测结果，此结果就是要用来填补空值的

    x_missing_reg.loc[x_missing_reg.iloc[:, i].isnull(), i] = y_predict
    f += 1
    if f > 1:
        break
