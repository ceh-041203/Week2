# 全数据集预测函数
def predict_all_samples(X, low_model, high_model, original_features, true_values):
    pred_low = low_model.predict(X)
    pred_high = high_model.predict(X)

    results_df = original_features.copy()
    results_df['True_Low_Price'] = true_values['Low Price'].values
    results_df['True_High_Price'] = true_values['High Price'].values
    results_df['Predicted_Low_Price'] = pred_low
    results_df['Predicted_High_Price'] = pred_high
    results_df['Low_Price_Diff'] = results_df['Predicted_Low_Price'] - results_df['True_Low_Price']
    results_df['High_Price_Diff'] = results_df['Predicted_High_Price'] - results_df['True_High_Price']

    return results_df