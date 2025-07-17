import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
# 随机森林模型自定义参数优化
def optimize_random_forest_custom(X_train, y_train, X_test, y_test, X, y, price_type="Low Price", cv_folds=5):
    print(f"\n===== 开始{price_type}随机森林自定义参数优化 =====")

    # 自定义参数组合
    param_combinations = [
        {'n_estimators': 20,
         'max_depth': 10,
         'min_samples_split': 6,
         'min_samples_leaf': 4}
    ]

    best_score = -np.inf  # 跟踪最佳R²分数
    best_model = None
    best_params = None
    results = []
    cv_results = None  # 存储最佳模型的交叉验证结果

    # 定义交叉验证折（与原始模型保持一致）
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for i, params in enumerate(param_combinations, 1):
        print(f"\n测试参数组合 {i}/{len(param_combinations)}: {params}")

        # 交叉验证评估
        fold_metrics = {
            'train_size': [],
            'val_size': [],
            'mse': [],
            'r2': []
        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            # 记录样本量
            fold_metrics['train_size'].append(len(train_idx))
            fold_metrics['val_size'].append(len(val_idx))

            # 训练模型
            rf = RandomForestRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_fold, y_train_fold)

            # 验证集评估
            y_pred = rf.predict(X_val_fold)
            mse = mean_squared_error(y_val_fold, y_pred)
            r2 = r2_score(y_val_fold, y_pred)
            fold_metrics['mse'].append(mse)
            fold_metrics['r2'].append(r2)
            print(f"  折 {fold}: 训练样本={len(train_idx)}, 验证样本={len(val_idx)}, MSE={mse:.4f}, R²={r2:.4f}")

        # 测试集评估（使用最佳参数的全量训练）
        rf_full = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        rf_full.fit(X_train, y_train)
        y_pred_test = rf_full.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)

        # 存储结果
        results.append({
            'params': params,
            'cv_mse': fold_metrics['mse'],
            'cv_r2': fold_metrics['r2'],
            'cv_train_size': fold_metrics['train_size'],
            'cv_val_size': fold_metrics['val_size'],
            'test_mse': test_mse,
            'test_r2': test_r2
        })

        print(f"\n  参数组合测试集评估: MSE = {test_mse:.4f}, R² = {test_r2:.4f}")

        # 更新最佳模型
        if test_r2 > best_score:
            best_score = test_r2
            best_model = rf_full
            best_params = params
            cv_results = fold_metrics  # 保存最佳模型的交叉验证结果

    # 输出最佳参数
    print(f"\n{price_type}最佳参数组合: {best_params}")
    print(f"{price_type}最佳模型交叉验证平均MSE: {np.mean(cv_results['mse']):.4f}")
    print(f"{price_type}最佳模型交叉验证平均R²: {np.mean(cv_results['r2']):.4f}")
    print(
        f"{price_type}最佳模型测试集MSE: {results[[res['params'] == best_params for res in results].index(True)]['test_mse']:.4f}")
    print(f"{price_type}最佳模型测试集R²: {best_score:.4f}")

    return (best_model, best_params,
            {'MSE': results[[res['params'] == best_params for res in results].index(True)]['test_mse'],
             'R²': best_score},
            cv_results)  # 返回交叉验证结果


# 训练多个原始模型（双目标预测）
def train_original_models(X_train, X_test, y_low_train, y_low_test, y_high_train, y_high_test, X, y_low, y_high):
    # 定义多个模型
    models = {
        'SVR': SVR(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'KNN': KNeighborsRegressor(),
        'XGBoost': xgb.XGBRegressor(random_state=42, objective='reg:squarederror'),
        'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1)
    }

    results = {}
    cv_folds = 5  # 交叉验证折数
    trained_models = {}  # 存储训练好的模型
    model_hyperparams = {}  # 存储模型超参数
    fold_sizes = {}  # 存储每折的样本数量

    # 预计算交叉验证折数的样本量（所有模型共享相同折划分）
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_sizes['train'] = []
    fold_sizes['val'] = []
    for train_idx, val_idx in kf.split(X):
        fold_sizes['train'].append(len(train_idx))
        fold_sizes['val'].append(len(val_idx))

    for model_name, model in models.items():
        print(f"\n===== 处理模型: {model_name} =====")
        model_results = {}  # 存储当前模型的所有结果
        model_hyperparams[model_name] = model.get_params()  # 记录超参数

        # 交叉验证评估（最低价格）
        print(f"\n【最低价格交叉验证 ({cv_folds}折)】")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_low), 1):
            X_train_fold = X.iloc[train_idx]
            y_low_train_fold = y_low.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_low_val_fold = y_low.iloc[val_idx]

            # 记录当前折的样本量
            model_results[f'Low_Fold_{fold}_Train_Size'] = len(train_idx)
            model_results[f'Low_Fold_{fold}_Val_Size'] = len(val_idx)

            try:
                model.fit(X_train_fold, y_low_train_fold)
                y_pred = model.predict(X_val_fold)
                mse = mean_squared_error(y_low_val_fold, y_pred)
                r2 = r2_score(y_low_val_fold, y_pred)
                model_results[f'Low_Fold {fold} MSE'] = mse
                model_results[f'Low_Fold {fold} R2'] = r2
                print(f"  折 {fold}: 训练样本={len(train_idx)}, 验证样本={len(val_idx)}, MSE={mse:.4f}, R²={r2:.4f}")
            except Exception as e:
                print(f"  折 {fold} 训练出错: {str(e)}")
                model_results[f'Low_Fold {fold} MSE'] = np.nan
                model_results[f'Low_Fold {fold} R2'] = np.nan

        # 交叉验证评估（最高价格）
        print(f"\n【最高价格交叉验证 ({cv_folds}折)】")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_high), 1):
            X_train_fold = X.iloc[train_idx]
            y_high_train_fold = y_high.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_high_val_fold = y_high.iloc[val_idx]

            # 记录当前折的样本量
            model_results[f'High_Fold_{fold}_Train_Size'] = len(train_idx)
            model_results[f'High_Fold_{fold}_Val_Size'] = len(val_idx)

            try:
                model.fit(X_train_fold, y_high_train_fold)
                y_pred = model.predict(X_val_fold)
                mse = mean_squared_error(y_high_val_fold, y_pred)
                r2 = r2_score(y_high_val_fold, y_pred)
                model_results[f'High_Fold {fold} MSE'] = mse
                model_results[f'High_Fold {fold} R2'] = r2
                print(f"  折 {fold}: 训练样本={len(train_idx)}, 验证样本={len(val_idx)}, MSE={mse:.4f}, R²={r2:.4f}")
            except Exception as e:
                print(f"  折 {fold} 训练出错: {str(e)}")
                model_results[f'High_Fold {fold} MSE'] = np.nan
                model_results[f'High_Fold {fold} R2'] = np.nan

        # 测试集评估（最低价格）
        try:
            low_model = clone(model)
            low_model.fit(X_train, y_low_train)
            y_pred = low_model.predict(X_test)
            test_mse = mean_squared_error(y_low_test, y_pred)
            test_r2 = r2_score(y_low_test, y_pred)
            model_results['Low_Test MSE'] = test_mse
            model_results['Low_Test R2'] = test_r2
            model_results['Low_Test_Size'] = len(X_test)  # 记录测试集样本量
            print(f"\n【最低价格测试集评估】")
            print(f"  最低价格测试集: 样本数={len(X_test)}, MSE = {test_mse:.4f}, R² = {test_r2:.4f}")
            trained_models[f'Original_{model_name}_LowPrice'] = low_model
        except Exception as e:
            print(f"  最低价格测试集评估出错: {str(e)}")
            model_results['Low_Test MSE'] = np.nan
            model_results['Low_Test R2'] = np.nan
            model_results['Low_Test_Size'] = len(X_test)

        # 测试集评估（最高价格）
        try:
            high_model = clone(model)
            high_model.fit(X_train, y_high_train)
            y_pred = high_model.predict(X_test)
            test_mse = mean_squared_error(y_high_test, y_pred)
            test_r2 = r2_score(y_high_test, y_pred)
            model_results['High_Test MSE'] = test_mse
            model_results['High_Test R2'] = test_r2
            model_results['High_Test_Size'] = len(X_test)  # 记录测试集样本量
            print(f"\n【最高价格测试集评估】")
            print(f"  最高价格测试集: 样本数={len(X_test)}, MSE = {test_mse:.4f}, R² = {test_r2:.4f}")
            trained_models[f'Original_{model_name}_HighPrice'] = high_model
        except Exception as e:
            print(f"  最高价格测试集评估出错: {str(e)}")
            model_results['High_Test MSE'] = np.nan
            model_results['High_Test R2'] = np.nan
            model_results['High_Test_Size'] = len(X_test)

        results[model_name] = model_results

    return pd.DataFrame(results).T, trained_models, model_hyperparams, fold_sizes