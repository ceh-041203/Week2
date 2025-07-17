import warnings
import os  # 用于文件夹操作
from sklearn.linear_model._cd_fast import ConvergenceWarning

from Dataset_forecasting import predict_all_samples
from Save_results import save_experiment_results
from data_preparation import prepare_datasets
from data_pretreatment import load_and_preprocess_data
from log import configure_logging
from model import train_original_models, optimize_random_forest_custom
from Random_forest_visualization import visualize_random_forest_tree

# 主函数
def find_best_models(results, optimized_results):
    pass


def main():
    configure_logging()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df, _, _, original_features, true_values = load_and_preprocess_data()
    X_train, X_test, y_low_train, y_low_test, y_high_train, y_high_test, X, y_low, y_high = prepare_datasets(df)

    # 训练原始模型（获取折样本量信息）
    results, trained_models, original_hyperparams, fold_sizes = train_original_models(
        X_train, X_test, y_low_train, y_low_test, y_high_train, y_high_test, X, y_low, y_high
    )
    # 记录测试集样本量
    fold_sizes['test_size'] = len(X_test)

    # 优化随机森林（添加交叉验证）
    optimized_models = {}
    optimized_results = {}
    optimized_cv_results = {}  # 存储优化模型的交叉验证结果

    # 优化最低价格模型（传入完整X和y用于交叉验证）
    opt_rf_low, low_params, low_metrics, low_cv = optimize_random_forest_custom(
        X_train, y_low_train, X_test, y_low_test,
        X, y_low,  # 完整数据集用于交叉验证
        price_type="Low Price"
    )
    optimized_models['Optimized_RandomForest_LowPrice'] = opt_rf_low
    optimized_results['Low Price'] = low_metrics
    optimized_cv_results['Low Price'] = low_cv  # 保存交叉验证结果

    # 优化最高价格模型
    opt_rf_high, high_params, high_metrics, high_cv = optimize_random_forest_custom(
        X_train, y_high_train, X_test, y_high_test,
        X, y_high,  # 完整数据集用于交叉验证
        price_type="High Price"
    )
    optimized_models['Optimized_RandomForest_HighPrice'] = opt_rf_high
    optimized_results['High Price'] = high_metrics
    optimized_cv_results['High Price'] = high_cv  # 保存交叉验证结果

    trained_models.update(optimized_models)

    # 分析最优模型
    best_models = find_best_models(results, optimized_results)

    # 打印对比结果
    print('\n_______________________________________________')
    print("\n各模型实验结果（包括优化后随机森林）：")
    comparison_df = results[['Low_Test MSE', 'Low_Test R2', 'High_Test MSE', 'High_Test R2']].copy()
    comparison_df.loc['Optimized_RandomForest'] = [
        optimized_results['Low Price']['MSE'],
        optimized_results['Low Price']['R²'],
        optimized_results['High Price']['MSE'],
        optimized_results['High Price']['R²']
    ]
    print(comparison_df.round(4))

    # 可视化决策树
    print("\n\n===== 随机森林决策树对比可视化 =====")
    if 'Original_RandomForest_LowPrice' in trained_models:
        visualize_random_forest_tree(trained_models['Original_RandomForest_LowPrice'], X_train.columns, "最低价格",
                                     "原始")
    if 'Optimized_RandomForest_LowPrice' in trained_models:
        visualize_random_forest_tree(trained_models['Optimized_RandomForest_LowPrice'], X_train.columns, "最低价格",
                                     "优化后")
    if 'Original_RandomForest_HighPrice' in trained_models:
        visualize_random_forest_tree(trained_models['Original_RandomForest_HighPrice'], X_train.columns, "最高价格",
                                     "原始")
    if 'Optimized_RandomForest_HighPrice' in trained_models:
        visualize_random_forest_tree(trained_models['Optimized_RandomForest_HighPrice'], X_train.columns, "最高价格",
                                     "优化后")

    # 模型映射
    model_mapping = {
        'Optimized_RandomForest': {'low': 'Optimized_RandomForest_LowPrice',
                                   'high': 'Optimized_RandomForest_HighPrice'},
        'RandomForest': {'low': 'Original_RandomForest_LowPrice', 'high': 'Original_RandomForest_HighPrice'},
        'XGBoost': {'low': 'Original_XGBoost_LowPrice', 'high': 'Original_XGBoost_HighPrice'},
        'LightGBM': {'low': 'Original_LightGBM_LowPrice', 'high': 'Original_LightGBM_HighPrice'},
        'GradientBoosting': {'low': 'Original_GradientBoosting_LowPrice',
                             'high': 'Original_GradientBoosting_HighPrice'},
        'Ridge': {'low': 'Original_Ridge_LowPrice', 'high': 'Original_Ridge_HighPrice'},
        'Lasso': {'low': 'Original_Lasso_LowPrice', 'high': 'Original_Lasso_HighPrice'},
        'KNN': {'low': 'Original_KNN_LowPrice', 'high': 'Original_KNN_HighPrice'},
        'SVR': {'low': 'Original_SVR_LowPrice', 'high': 'Original_SVR_HighPrice'}
    }

    # 保存预测结果
    output_dir = "pumpkin_price_predictions"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n已创建预测结果文件夹: {output_dir}")
    else:
        print(f"\n使用现有预测结果文件夹: {output_dir}")

    print("\n\n===== 开始保存各模型全数据集预测结果 =====")
    for model_name in model_mapping.keys():
        try:
            low_model = trained_models[model_mapping[model_name]['low']]
            high_model = trained_models[model_mapping[model_name]['high']]
            prediction_results = predict_all_samples(X, low_model, high_model, original_features, true_values)
            file_path = os.path.join(output_dir, f'pumpkin_price_prediction_{model_name}.csv')
            prediction_results.to_csv(file_path, index=False)
            print(f"已保存 {model_name} 模型预测结果到: {file_path}")
        except Exception as e:
            print(f"保存 {model_name} 模型预测结果时出错: {str(e)}")

    # 保存实验结果（包含折样本量）
    save_experiment_results(results, original_hyperparams, optimized_results,
                            low_params, high_params, fold_sizes, output_dir, optimized_cv_results)

    print("\n所有模型预测结果和实验结果保存完成！")


if __name__ == "__main__":
    main()