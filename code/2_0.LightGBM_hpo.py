import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import os
import math
import optuna
import json

os.chdir("D:/정리/코드 포토폴리오/LightGBM_O3_factor")
print(os.getcwd())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./mid_result/O3_factor_SMA.csv")
    parser.add_argument("--target", type=str, default="O3_unit")
    parser.add_argument("--features", nargs="+", default=["RH", "u10", "v10", "SSR","HCHO", "NO2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_path", type=str, default="./mid_result/LightGBM_hpo/")
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--n_fold", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=50)

    # 빠져있던 옵션 추가
    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    parser.add_argument("--test_size", type=float, default=0.2)

    args = parser.parse_args()
    return args

def objective(trial, X, y, args, n_fold=10):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "random_state": args.seed,
        "n_jobs": -1,
        "verbose": -1,
        "device": "gpu" if args.use_gpu else "cpu",
    }

    kf = KFold(n_splits=n_fold, shuffle=True, random_state=args.seed)
    r2_scores, rmse_scores, mae_scores = [], [], []

    for train_idx, valid_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
        )

        y_pred = model.predict(X_val)
        r2_scores.append(r2_score(y_val, y_pred))
        rmse_scores.append(math.sqrt(mean_squared_error(y_val, y_pred)))
        mae_scores.append(mean_absolute_error(y_val, y_pred))

    return np.mean(r2_scores), np.mean(rmse_scores), np.mean(mae_scores)

def run_optuna_full(X, y, args):
    sampler = optuna.samplers.NSGAIISampler(seed=args.seed)
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        sampler=sampler
    )

    with tqdm(total=args.n_trials, desc=f"Optuna (Full {args.n_fold}-Fold)") as pbar:

        def callback(study_obj, trial_obj):
            pbar.update(1)
            vals = trial_obj.values
            pbar.set_postfix_str(f"R²={vals[0]:.4f}, RMSE={vals[1]:.4f}, MAE={vals[2]:.4f}")

        study.optimize(
            lambda t: objective(t, X, y, args, n_fold=args.n_fold),
            n_trials=args.n_trials,
            callbacks=[callback]
        )

    df_trials = study.trials_dataframe()
    df_trials.to_csv(os.path.join(args.results_path, "optuna_full_results_SMA.csv"), index=False)

    for t in study.best_trials:
        print(f"Trial {t.number}: R²={t.values[0]:.4f}, RMSE={t.values[1]:.4f}, MAE={t.values[2]:.4f}")
        print(f"  Params: {t.params}\n")

    return study

def select_best_balance(study):
    best, best_score = None, -1e9
    for t in study.best_trials:
        score = t.values[0] * 100 - (t.values[1] + t.values[2])
        if score > best_score:
            best, best_score = t, score
    return best

def save_best_params(best_trial, path):
    with open(path, "w") as f:
        json.dump(best_trial.params, f, indent=2)
def main():
    args = get_args()
    os.makedirs(args.results_path, exist_ok=True)

    df = pd.read_csv(args.data_path)
    X = df[args.features]
    y = df[args.target]

    full_study = run_optuna_full(X, y, args)
    best_trial = select_best_balance(full_study)

    print(f"R²={best_trial.values[0]:.4f}, RMSE={best_trial.values[1]:.4f}, MAE={best_trial.values[2]:.4f}")
    print(f"Params: {best_trial.params}")
    best_params_path = os.path.join(args.results_path, "best_params.json")
    save_best_params(best_trial, best_params_path)

    fixed_trial = optuna.trial.FixedTrial(best_trial.params)
    r2, rmse, mae = objective(fixed_trial, X, y, args, n_fold=args.n_fold)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    model = lgb.LGBMRegressor(**best_trial.params, random_state=args.seed, n_jobs=-1)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False)],
    )

    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)

    print("best testing performance:")
    print(f"R²   : {test_r2:.4f}")
    print(f"RMSE : {test_rmse:.4f}")
    print(f"MAE  : {test_mae:.4f}")

    results_df = pd.DataFrame([{
        **best_trial.params,
        "CV_R2": r2,
        "CV_RMSE": rmse,
        "CV_MAE": mae,
        "Test_R2": test_r2,
        "Test_RMSE": test_rmse,
        "Test_MAE": test_mae,
    }])

    results_df.to_csv(os.path.join(args.results_path, "lightgbm_final_result_SMA.csv"), index=False)


if __name__ == "__main__":
    main()
