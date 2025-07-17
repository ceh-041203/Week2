import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
# 可视化随机森林中的决策树
def visualize_random_forest_tree(trained_model, feature_names, price_type="最低价格", model_type="原始"):
    tree_to_visualize = trained_model.estimators_[0]
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.figure(figsize=(20, 12))
    plot_tree(
        tree_to_visualize,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        proportion=False,
        precision=2,
        fontsize=10
    )
    plt.title(f"{model_type}随机森林决策树可视化 ({price_type}模型)", fontsize=15)
    plt.tight_layout()
    plt.show()


# 分析结果并找出最优模型
def find_best_models(results_df, optimized_results=None):
    test_results = results_df[['Low_Test MSE', 'Low_Test R2', 'High_Test MSE', 'High_Test R2']].copy()
    if optimized_results:
        test_results.loc['Optimized_RandomForest'] = [
            optimized_results['Low Price']['MSE'],
            optimized_results['Low Price']['R²'],
            optimized_results['High Price']['MSE'],
            optimized_results['High Price']['R²']
        ]

    best_low_mse_model = test_results['Low_Test MSE'].idxmin()
    best_low_r2_model = test_results['Low_Test R2'].idxmax()
    best_high_mse_model = test_results['High_Test MSE'].idxmin()
    best_high_r2_model = test_results['High_Test R2'].idxmax()

    print("\n\n===== 最优模型分析 =====")
    print("\n最低价格预测:")
    print(f"最小MSE模型: {best_low_mse_model} (MSE = {test_results.loc[best_low_mse_model, 'Low_Test MSE']:.4f})")
    print(f"最大R²模型: {best_low_r2_model} (R² = {test_results.loc[best_low_r2_model, 'Low_Test R2']:.4f})")
    print("\n最高价格预测:")
    print(f"最小MSE模型: {best_high_mse_model} (MSE = {test_results.loc[best_high_mse_model, 'High_Test MSE']:.4f})")
    print(f"最大R²模型: {best_high_r2_model} (R² = {test_results.loc[best_high_r2_model, 'High_Test R2']:.4f})")

    return {
        'low_price': {'model': best_low_r2_model, 'r2': test_results.loc[best_low_r2_model, 'Low_Test R2'],
                      'mse': test_results.loc[best_low_r2_model, 'Low_Test MSE']},
        'high_price': {'model': best_high_r2_model, 'r2': test_results.loc[best_high_r2_model, 'High_Test R2'],
                       'mse': test_results.loc[best_high_r2_model, 'High_Test MSE']}
    }
