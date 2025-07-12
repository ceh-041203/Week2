#数据读取
import pandas as pd
df=pd.read_csv("US-pumpkins.csv")
print(df.info())
print('_______________________________________________')
#数据预处理
#1.统计每列的空缺值数目
missing_values = df.isnull().sum()
print('数据预处理前各列缺失值数')
print(missing_values)
print('_______________________________________________')
#2.删除全部为空白的列
columns_to_drop = missing_values[missing_values == df.shape[0]].index
df = df.drop(columns_to_drop, axis=1)
#3.删除无意义的列（空缺值>90%）
columns = df.columns
columns_to_drop = [col for col in ['Type', 'Sub Variety',
                                   'Origin District',
                                   'Unit of Sale',
                                   'Unnamed: 25'] if col in columns]
df = df.drop(columns_to_drop, axis=1)
#4.缺失值填充
# 定义分类变量和数值变量列表
categorical_cols = ['Variety', 'Origin', 'Item Size', 'Color']
numeric_cols = ['Mostly Low', 'Mostly High']
# 对分类变量使用众数填充缺失值
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
# 对数值变量使用中位数填充缺失值
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)
#5.处理后数据
print('数据预处理后各列缺失值数')
missing_values = df.isnull().sum()
print(missing_values)
print('_______________________________________________')
