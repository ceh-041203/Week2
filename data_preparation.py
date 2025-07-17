from sklearn.model_selection import train_test_split
import pandas as pd
# 数据准备（同时处理最低价格和最高价格）
def prepare_datasets(df):
    # 生成特征并处理特征名中的空格
    X = pd.get_dummies(df[['City Name', 'Package', 'Variety', 'Date',
                           'Origin', 'Item Size', 'Color', 'Repack']])
    X.columns = X.columns.str.replace(' ', '_')  # 处理特征名中的空格

    # 同时保留最低价格和最高价格作为目标变量
    y_low = df['Low Price']
    y_high = df['High Price']

    # 划分训练集和测试集
    X_train, X_test, y_low_train, y_low_test, y_high_train, y_high_test = train_test_split(
        X, y_low, y_high, test_size=0.2, random_state=42
    )

    print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    print('_______________________________________________\n')

    return X_train, X_test, y_low_train, y_low_test, y_high_train, y_high_test, X, y_low, y_high


