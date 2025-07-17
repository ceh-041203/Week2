import pandas as pd
import numpy as np
import os  # 用于文件夹操作
# 收集并保存所有模型实验结果
def save_experiment_results(original_results, original_hyperparams, optimized_results,
                            low_params, high_params, fold_sizes, output_dir, optimized_cv_results=None):
    all_results = []
    cv_folds = 5

    # 处理原始模型结果（代码保持不变）
    for model_name in original_results.index:
        result_row = original_results.loc[model_name]
        model_result = {
            '模型名称': f'Original_{model_name}',
            '超参数': str(original_hyperparams[model_name]),
            **{f'最低价格_折{i}_训练样本数': result_row.get(f'Low_Fold_{i}_Train_Size', np.nan) for i in
               range(1, cv_folds + 1)},
            **{f'最低价格_折{i}_验证样本数': result_row.get(f'Low_Fold_{i}_Val_Size', np.nan) for i in
               range(1, cv_folds + 1)},
            **{f'最低价格_折{i}_MSE': result_row.get(f'Low_Fold {i} MSE', np.nan) for i in range(1, cv_folds + 1)},
            **{f'最低价格_折{i}_R2': result_row.get(f'Low_Fold {i} R2', np.nan) for i in range(1, cv_folds + 1)},
            **{f'最高价格_折{i}_训练样本数': result_row.get(f'High_Fold_{i}_Train_Size', np.nan) for i in
               range(1, cv_folds + 1)},
            **{f'最高价格_折{i}_验证样本数': result_row.get(f'High_Fold_{i}_Val_Size', np.nan) for i in
               range(1, cv_folds + 1)},
            **{f'最高价格_折{i}_MSE': result_row.get(f'High_Fold {i} MSE', np.nan) for i in range(1, cv_folds + 1)},
            **{f'最高价格_折{i}_R2': result_row.get(f'High_Fold {i} R2', np.nan) for i in range(1, cv_folds + 1)},
            '最低价格_测试样本数': result_row.get('Low_Test_Size', np.nan),
            '最低价格_测试MSE': result_row.get('Low_Test MSE', np.nan),
            '最低价格_测试R2': result_row.get('Low_Test R2', np.nan),
            '最高价格_测试样本数': result_row.get('High_Test_Size', np.nan),
            '最高价格_测试MSE': result_row.get('High_Test MSE', np.nan),
            '最高价格_测试R2': result_row.get('High_Test R2', np.nan)
        }
        all_results.append(model_result)

    # 处理优化后随机森林结果（填充交叉验证数据）
    if optimized_cv_results:
        # 提取最低/最高价格的交叉验证结果
        low_cv = optimized_cv_results.get('Low Price', {})
        high_cv = optimized_cv_results.get('High Price', {})

        optimized_result = {
            '模型名称': 'Optimized_RandomForest',
            '超参数': str({'Low_Price_Params': low_params, 'High_Price_Params': high_params}),
            # 填充最低价格交叉验证结果
            **{f'最低价格_折{i}_训练样本数': low_cv.get('train_size', [np.nan] * cv_folds)[i - 1] for i in
               range(1, cv_folds + 1)},
            **{f'最低价格_折{i}_验证样本数': low_cv.get('val_size', [np.nan] * cv_folds)[i - 1] for i in
               range(1, cv_folds + 1)},
            **{f'最低价格_折{i}_MSE': low_cv.get('mse', [np.nan] * cv_folds)[i - 1] for i in range(1, cv_folds + 1)},
            **{f'最低价格_折{i}_R2': low_cv.get('r2', [np.nan] * cv_folds)[i - 1] for i in range(1, cv_folds + 1)},
            # 填充最高价格交叉验证结果
            **{f'最高价格_折{i}_训练样本数': high_cv.get('train_size', [np.nan] * cv_folds)[i - 1] for i in
               range(1, cv_folds + 1)},
            **{f'最高价格_折{i}_验证样本数': high_cv.get('val_size', [np.nan] * cv_folds)[i - 1] for i in
               range(1, cv_folds + 1)},
            **{f'最高价格_折{i}_MSE': high_cv.get('mse', [np.nan] * cv_folds)[i - 1] for i in range(1, cv_folds + 1)},
            **{f'最高价格_折{i}_R2': high_cv.get('r2', [np.nan] * cv_folds)[i - 1] for i in range(1, cv_folds + 1)},
            # 测试集结果
            '最低价格_测试样本数': original_results.iloc[0]['Low_Test_Size'] if not original_results.empty else np.nan,
            '最低价格_测试MSE': optimized_results['Low Price']['MSE'],
            '最低价格_测试R2': optimized_results['Low Price']['R²'],
            '最高价格_测试样本数': original_results.iloc[0]['High_Test_Size'] if not original_results.empty else np.nan,
            '最高价格_测试MSE': optimized_results['High Price']['MSE'],
            '最高价格_测试R2': optimized_results['High Price']['R²']
        }
    else:
        # 兼容无交叉验证结果的情况（默认NaN）
        optimized_result = {
            '模型名称': 'Optimized_RandomForest',
            '超参数': str({'Low_Price_Params': low_params, 'High_Price_Params': high_params}),
            **{f'最低价格_折{i}_训练样本数': fold_sizes['train'][i - 1] for i in range(1, cv_folds + 1)},
            **{f'最低价格_折{i}_验证样本数': fold_sizes['val'][i - 1] for i in range(1, cv_folds + 1)},
            **{f'最低价格_折{i}_MSE': np.nan for i in range(1, cv_folds + 1)},
            **{f'最低价格_折{i}_R2': np.nan for i in range(1, cv_folds + 1)},
            **{f'最高价格_折{i}_训练样本数': fold_sizes['train'][i - 1] for i in range(1, cv_folds + 1)},
            **{f'最高价格_折{i}_验证样本数': fold_sizes['val'][i - 1] for i in range(1, cv_folds + 1)},
            **{f'最高价格_折{i}_MSE': np.nan for i in range(1, cv_folds + 1)},
            **{f'最高价格_折{i}_R2': np.nan for i in range(1, cv_folds + 1)},
            '最低价格_测试样本数': original_results.iloc[0]['Low_Test_Size'] if not original_results.empty else np.nan,
            '最低价格_测试MSE': optimized_results['Low Price']['MSE'],
            '最低价格_测试R2': optimized_results['Low Price']['R²'],
            '最高价格_测试样本数': original_results.iloc[0]['High_Test_Size'] if not original_results.empty else np.nan,
            '最高价格_测试MSE': optimized_results['High Price']['MSE'],
            '最高价格_测试R2': optimized_results['High Price']['R²']
        }

    all_results.append(optimized_result)

    # 保存结果（关键修改：调整列顺序逻辑）
    results_df = pd.DataFrame(all_results)

    # 按折数分组排列列（最低价格→最高价格，每个折内按指标类型排列）
    columns_order = ['模型名称', '超参数']

    # 最低价格：按折数循环，每个折内按"训练样本数→验证样本数→MSE→R2"排列
    for i in range(1, cv_folds + 1):
        columns_order.extend([
            f'最低价格_折{i}_训练样本数',
            f'最低价格_折{i}_验证样本数',
            f'最低价格_折{i}_MSE',
            f'最低价格_折{i}_R2'
        ])

    # 最高价格：按折数循环，每个折内按"训练样本数→验证样本数→MSE→R2"排列
    for i in range(1, cv_folds + 1):
        columns_order.extend([
            f'最高价格_折{i}_训练样本数',
            f'最高价格_折{i}_验证样本数',
            f'最高价格_折{i}_MSE',
            f'最高价格_折{i}_R2'
        ])

    # 测试集结果
    columns_order.extend([
        '最低价格_测试样本数', '最低价格_测试MSE', '最低价格_测试R2',
        '最高价格_测试样本数', '最高价格_测试MSE', '最高价格_测试R2'
    ])

    # 强制按指定顺序排列列
    results_df = results_df[columns_order]

    output_path = os.path.join(output_dir, 'model_experiment_results.csv')
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n已保存所有模型实验结果到: {output_path}")
    return results_df