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