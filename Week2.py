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

#预测建模
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 对字符型特征进行独热编码
X = pd.get_dummies(df[['City Name', 'Package', 'Variety', 'Date', 'Origin', 'Item Size', 'Color', 'Repack']])

# 最低价格和最高价格
y_low = df['Low Price']
y_high = df['High Price']

# 划分训练集和测试集
X_train, X_test, y_low_train, y_low_test, y_high_train, y_high_test = train_test_split(X, y_low, y_high,
                                                                                       test_size=0.2,
                                                                                       random_state=42)

# 定义模型
models = {
    'SVR': SVR(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'KNN': KNeighborsRegressor()
}

# 用于存储每个模型的评估结果
results = {}

for model_name, model in models.items():
    # 训练最低价格预测模型
    model.fit(X_train, y_low_train)
    y_low_pred = model.predict(X_test)

    # 训练最高价格预测模型
    model.fit(X_train, y_high_train)
    y_high_pred = model.predict(X_test)

    # 计算评估指标
    mse_low = mean_squared_error(y_low_test, y_low_pred)
    mse_high = mean_squared_error(y_high_test, y_high_pred)
    r2_low = r2_score(y_low_test, y_low_pred)
    r2_high = r2_score(y_high_test, y_high_pred)

    # 存储结果
    results[model_name] = {
        'Low Price MSE': mse_low,
        'High Price MSE': mse_high,
        'Low Price R2': r2_low,
        'High Price R2': r2_high
    }

# 将结果转换为 DataFrame 以便更好地展示
results_df = pd.DataFrame(results).T

# 可视化
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['SimHei']
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 绘制最低价格 MSE 柱状图
axes[0, 0].bar(results_df.index, results_df['Low Price MSE'])
axes[0, 0].set_title('最低价格均方误差（MSE）')
axes[0, 0].set_xlabel('模型')
axes[0, 0].set_ylabel('MSE')

# 为最低价格 MSE 柱状图添加数值标签
for i, v in enumerate(results_df['Low Price MSE']):
    axes[0, 0].text(i, v, str(round(v, 2)), ha='center', va='bottom')

# 绘制最高价格 MSE 柱状图
axes[0, 1].bar(results_df.index, results_df['High Price MSE'])
axes[0, 1].set_title('最高价格均方误差（MSE）')
axes[0, 1].set_xlabel('模型')
axes[0, 1].set_ylabel('MSE')

# 为最高价格 MSE 柱状图添加数值标签
for i, v in enumerate(results_df['High Price MSE']):
    axes[0, 1].text(i, v, str(round(v, 2)), ha='center', va='bottom')

# 绘制最低价格 R2 分数柱状图
axes[1, 0].bar(results_df.index, results_df['Low Price R2'])
axes[1, 0].set_title('最低价格 R2 分数')
axes[1, 0].set_xlabel('模型')
axes[1, 0].set_ylabel('R2 分数')

# 为最低价格 R2 分数柱状图添加数值标签
for i, v in enumerate(results_df['Low Price R2']):
    axes[1, 0].text(i, v, str(round(v, 2)), ha='center', va='bottom')

# 绘制最高价格 R2 分数柱状图
axes[1, 1].bar(results_df.index, results_df['High Price R2'])
axes[1, 1].set_title('最高价格 R2 分数')
axes[1, 1].set_xlabel('模型')
axes[1, 1].set_ylabel('R2 分数')

# 为最高价格 R2 分数柱状图添加数值标签
for i, v in enumerate(results_df['High Price R2']):
    axes[1, 1].text(i, v, str(round(v, 2)), ha='center', va='bottom')

plt.tight_layout()
plt.show()
print(results_df)
print('_______________________________________________')

#部分模型优化
import warnings
from sklearn.linear_model._cd_fast import ConvergenceWarning
# 忽略特定的收敛警告
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.linear_model._coordinate_descent")

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# 定义要优化的模型及其参数网格
models_params = {
    'SVR': (SVR(), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }),
    'Lasso': (Lasso(random_state=42), {
        'alpha': [0.001, 0.01, 0.1, 1]
    })
}

# 用于存储每个模型的评估结果
results = {}

for model_name, (model, param_grid) in models_params.items():
    # 使用网格搜索进行超参数调优
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
    # 训练最低价格预测模型
    grid_search.fit(X_train, y_low_train)
    best_model_low = grid_search.best_estimator_
    y_low_pred = best_model_low.predict(X_test)

    # 训练最高价格预测模型
    grid_search.fit(X_train, y_high_train)
    best_model_high = grid_search.best_estimator_
    y_high_pred = best_model_high.predict(X_test)

    # 计算评估指标
    mse_low = mean_squared_error(y_low_test, y_low_pred)
    mse_high = mean_squared_error(y_high_test, y_high_pred)
    r2_low = r2_score(y_low_test, y_low_pred)
    r2_high = r2_score(y_high_test, y_high_pred)

    # 存储结果
    results[model_name] = {
        'Low Price MSE': mse_low,
        'High Price MSE': mse_high,
        'Low Price R2': r2_low,
        'High Price R2': r2_high
    }


results_df = pd.DataFrame(results).T

print("部分优化后模型效果：")
print(results_df)
print('_______________________________________________')