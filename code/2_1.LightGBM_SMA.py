import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import os
import math
from matplotlib import font_manager
import matplotlib.pyplot as plt

path = './data/TimesNewerRoman-Regular.otf'
font_manager.fontManager.addfont(path)
font_name = font_manager.FontProperties(fname=path).get_name()
plt.rcParams['font.family'] = font_name

os.chdir("D:/정리/코드 포토폴리오/LightGBM_O3_factor")
print(os.getcwd())

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--best_params_path",
        type=str,
        default="./mid_result/LightGBM_hpo/best_params.json"
    )

    args, _ = parser.parse_known_args()

    best_params = {}
    if os.path.exists(args.best_params_path):
        import json
        with open(args.best_params_path) as f:
            best_params = json.load(f)
        print(f"Loaded best params from {args.best_params_path}")

    parser.add_argument("--n_estimators", type=int,
                        default=best_params.get("n_estimators", 1000))
    parser.add_argument("--learning_rate", type=float,
                        default=best_params.get("learning_rate", 0.05))
    parser.add_argument("--max_depth", type=int,
                        default=best_params.get("max_depth", -1))
    parser.add_argument("--num_leaves", type=int,
                        default=best_params.get("num_leaves", 31))
    parser.add_argument("--min_child_samples", type=int,
                        default=best_params.get("min_child_samples", 20))
    parser.add_argument("--subsample", type=float,
                        default=best_params.get("subsample", 1.0))
    parser.add_argument("--colsample_bytree", type=float,
                        default=best_params.get("colsample_bytree", 1.0))
    parser.add_argument("--reg_alpha", type=float,
                        default=best_params.get("reg_alpha", 0.0))
    parser.add_argument("--reg_lambda", type=float,
                        default=best_params.get("reg_lambda", 0.0))

    parser.add_argument("--data_path", type=str,
                        default="./mid_result/O3_factor_SMA.csv")
    parser.add_argument("--target", type=str, default="O3_unit")
    parser.add_argument("--features", nargs="+",
                        default=["RH", "u10", "v10", "SSR", "HCHO", "NO2"])
    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_path", type=str, default="./LightGBM_result/")

    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--verbosity", type=int, default=-1)
    parser.add_argument("--early_stopping_rounds", type=int, default=100)
    parser.add_argument("--n_fold", type=int, default=10)

    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.results_path, exist_ok=True)

    df = pd.read_csv(args.data_path)

    X = df[args.features]
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    kf = KFold(n_splits=args.n_fold, shuffle=True, random_state=args.seed)

    cv_r2_list, cv_rmse_list, cv_mae_list = [], [], []

    lgb_params = dict(
        objective="regression",
        boosting_type="gbdt",
        metric="rmse",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        verbosity=-1,
        seed=args.seed,
        num_threads=args.n_jobs,
        device_type="gpu" if args.use_gpu else "cpu"
    )

    for train_idx, valid_idx in tqdm(
        kf.split(X), total=args.n_fold, desc=f"{args.n_fold}-Fold CV"
    ):
        X_train_cv, X_valid_cv = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_cv, y_valid_cv = y.iloc[train_idx], y.iloc[valid_idx]

        train_data = lgb.Dataset(X_train_cv, label=y_train_cv)
        valid_data = lgb.Dataset(X_valid_cv, label=y_valid_cv)

        model_cv = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)]
        )

        y_pred_cv = model_cv.predict(X_valid_cv)

        cv_r2_list.append(r2_score(y_valid_cv, y_pred_cv))
        cv_rmse_list.append(math.sqrt(mean_squared_error(y_valid_cv, y_pred_cv)))
        cv_mae_list.append(mean_absolute_error(y_valid_cv, y_pred_cv))

    print("CV Performance:")
    print(f"CV Mean R²   : {np.mean(cv_r2_list):.4f}")
    print(f"CV Mean RMSE : {np.mean(cv_rmse_list):.4f}")
    print(f"CV Mean MAE  : {np.mean(cv_mae_list):.4f}")


    final_model = lgb.LGBMRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=args.verbosity,
        device="gpu" if args.use_gpu else "cpu",
    )

    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
    )

    y_pred = final_model.predict(X_test)

    test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    print("\nModel Training Performance:")
    print(f"Test RMSE : {test_rmse:.4f}")
    print(f"Test R²   : {test_r2:.4f}")
    print(f"Test MAE  : {test_mae:.4f}")
    
    test_result_path = os.path.join(args.results_path, "test_performance_SMA.txt")
    
    with open(test_result_path, "w", encoding="utf-8") as f:
        f.write(f"Test RMSE : {test_rmse:.4f}\n")
        f.write(f"Test R²   : {test_r2:.4f}\n")
        f.write(f"Test MAE  : {test_mae:.4f}\n")


    from sklearn.inspection import PartialDependenceDisplay

    print("\nPDP start")

    for feature in ["NO2", "HCHO"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        display = PartialDependenceDisplay.from_estimator(
            final_model, X_train, features=[feature], ax=ax
        )

        feature_label = "NO$_2$" if feature == "NO2" else "HCHO"
        display.axes_[0, 0].set_xlabel(
            f"{feature_label} ($10^{{15}}$ molec/cm$^2$)", fontsize=13
        )
        display.axes_[0, 0].set_ylabel(
            "Predicted O$_3$ $(µg/m^3)$", fontsize=13
        )

        plt.title(f"Partial Dependence of {feature_label} on O$_3$ \nin SMA", fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(args.results_path, f"pdp_{feature}_SMA.png"), dpi=300)
        plt.close()


    import shap

    print("\nSHAP start")

    rename_map = {
        "u10": "U10",
        "v10": "V10",
        "NO2": "NO$_2$",
        "HCHO": "HCHO"
    }
    X_train_shap = X_train.rename(columns=rename_map)

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_train_shap)

    shap.summary_plot(shap_values, X_train_shap, show=False)
    plt.xlabel("SHAP value (impact on model output)", fontsize=15, fontweight='bold')
    plt.title("SHAP Summary Plot(SMA)", fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(args.results_path, "shap_summary_SMA.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
