import pandas as pd
# 数据加载与预处理
def load_and_preprocess_data(file_path="US-pumpkins.csv"):
    df = pd.read_csv(file_path)
    print("数据基本信息：")
    print(df.info())
    print('_______________________________________________')

    # 保存原始特征数据
    original_features = df[['City Name', 'Package', 'Variety', 'Date',
                            'Origin', 'Item Size', 'Color', 'Repack',
                            'Mostly Low', 'Mostly High']].copy()

    # 统计缺失值
    missing_values = df.isnull().sum()
    print('数据预处理前各列缺失值数')
    print(missing_values)
    print('_______________________________________________')

    # 保存原始数据索引和真实值，用于后续完整预测
    true_values = df[['Low Price', 'High Price']].copy()

    # 删除全空列和高缺失值列
    columns_to_drop = missing_values[missing_values == df.shape[0]].index
    df = df.drop(columns_to_drop, axis=1)

    columns = df.columns
    columns_to_drop = [col for col in ['Type', 'Sub Variety',
                                       'Origin District', 'Unit of Sale', 'Unnamed: 25'] if col in columns]
    df = df.drop(columns_to_drop, axis=1)

    # 缺失值填充
    categorical_cols = ['Variety', 'Origin', 'Item Size', 'Color']
    numeric_cols = ['Mostly Low', 'Mostly High']

    # 保存填充值用于特征处理
    cat_fill_values = {}
    num_fill_values = {}

    for col in categorical_cols:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            cat_fill_values[col] = mode_val
    for col in numeric_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            num_fill_values[col] = median_val

    # 预处理后缺失值检查
    print('数据预处理后各列缺失值数')
    missing_values = df.isnull().sum()
    print(missing_values)
    print('_______________________________________________')

    return df, cat_fill_values, num_fill_values, original_features, true_values
