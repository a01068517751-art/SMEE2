# train.py
# 목적:
# 기존 main.py 알고리즘의 출력 p_base에 대해
# residual correction용 Ridge 상수를 학습한다.
#
# 사용 방법:
# 1) 같은 폴더에 main.py가 있어야 함
# 2) data_set/train.mat, data_set/val.mat 이 있으면 그것을 사용
# 3) 없으면 data_set/InF_DH_FR1.mat 을 500/100으로 자동 분할해서 사용
# 4) 실행 후 출력되는 RIDGE_* 상수를 main.py에 복붙

import numpy as np
import scipy.io as sio
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = PROJECT_ROOT / "main"

if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from main5 import (
    your_algorithm,
    make_ridge_feature,
    rmse,
    mae,
    LOWER_BOUND,
    UPPER_BOUND,
    EPS
)


# ============================================================
# 데이터 로드 유틸
# ============================================================

def get_p_bs(data):
    if "p_bs" in data:
        return np.asarray(data["p_bs"], dtype=float)
    if "BS_positions" in data:
        return np.asarray(data["BS_positions"], dtype=float)
    raise KeyError("p_bs 또는 BS_positions 변수를 찾을 수 없습니다.")


def load_train_val_data():
    data_dir = PROJECT_ROOT / "data_set"

    train_path = data_dir / "train.mat"
    val_path = data_dir / "val.mat"

    if train_path.exists() and val_path.exists():
        print("========== Load train.mat / val.mat ==========")

        train_data = sio.loadmat(train_path, squeeze_me=False)
        val_data = sio.loadmat(val_path, squeeze_me=False)

        p_train = np.asarray(train_data["p"], dtype=float)
        d_train = np.asarray(train_data["d_hat"], dtype=float)

        p_val = np.asarray(val_data["p"], dtype=float)
        d_val = np.asarray(val_data["d_hat"], dtype=float)

        p_bs = get_p_bs(train_data)

        return p_train, d_train, p_val, d_val, p_bs

    full_path = data_dir / "InF_DH_FR1.mat"

    if not full_path.exists():
        raise FileNotFoundError(
            "data_set/train.mat, data_set/val.mat 또는 data_set/InF_DH_FR1.mat 파일을 찾을 수 없습니다."
        )

    print("========== train.mat / val.mat 없음 ==========")
    print("========== InF_DH_FR1.mat에서 500/100 자동 분할 ==========")

    full_data = sio.loadmat(full_path, squeeze_me=False)

    p_all = np.asarray(full_data["p"], dtype=float)
    d_all = np.asarray(full_data["d_hat"], dtype=float)
    p_bs = get_p_bs(full_data)

    num_user = d_all.shape[1]

    if num_user < 600:
        raise ValueError("자동 분할을 위해서는 최소 600개 이상의 데이터가 필요합니다.")

    # 고정 seed로 재현 가능하게 분할
    rng = np.random.default_rng(42)
    idx = np.arange(num_user)
    rng.shuffle(idx)

    train_idx = idx[:500]
    val_idx = idx[500:600]

    p_train = p_all[:, train_idx]
    d_train = d_all[:, train_idx]

    p_val = p_all[:, val_idx]
    d_val = d_all[:, val_idx]

    return p_train, d_train, p_val, d_val, p_bs


# ============================================================
# Ridge closed-form 학습
# ============================================================

def fit_ridge_closed_form(X, Y, alpha):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    mean = X.mean(axis=0)
    std = X.std(axis=0) + EPS

    Xn = (X - mean) / std

    # intercept 항 추가
    X_aug = np.hstack([
        Xn,
        np.ones((Xn.shape[0], 1), dtype=float)
    ])

    # Ridge closed-form:
    # W = inv(X^T X + alpha I) X^T Y
    I = np.eye(X_aug.shape[1], dtype=float)
    I[-1, -1] = 0.0  # intercept에는 regularization 적용 안 함

    A = X_aug.T @ X_aug + alpha * I
    B = X_aug.T @ Y

    W = np.linalg.solve(A, B)

    coef = W[:-1, :]
    intercept = W[-1, :]

    return mean, std, coef, intercept


def predict_ridge_delta(X, mean, std, coef, intercept, clip):
    X = np.asarray(X, dtype=float)

    Xn = (X - mean) / (std + EPS)

    delta = Xn @ coef + intercept
    delta = np.asarray(delta, dtype=float)

    delta = np.clip(delta, -clip, clip)

    return delta


# ============================================================
# 기존 알고리즘 결과 p_base와 feature/target 생성
# ============================================================

def build_feature_target(d_hat, p_true, p_bs, name="dataset"):
    num_user = d_hat.shape[1]

    X = []
    Y = []
    P_BASE = np.zeros((2, num_user), dtype=float)

    print(f"\n========== Build Features: {name} ==========")

    for u in range(num_user):
        d_u = d_hat[:, u]

        p_base = your_algorithm(
            d_raw=d_u,
            p_bs=p_bs
        )

        feat = make_ridge_feature(
            p_base=p_base,
            d_raw=d_u,
            p_bs=p_bs
        )

        delta_true = p_true[:, u] - p_base

        X.append(feat)
        Y.append(delta_true)
        P_BASE[:, u] = p_base

        if (u + 1) % 50 == 0 or (u + 1) == num_user:
            print(f"{name}: {u + 1}/{num_user} done")

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    return X, Y, P_BASE


# ============================================================
# 보정 적용 후 평가
# ============================================================

def evaluate_with_correction(P_BASE, X, p_true, mean, std, coef, intercept, lam, clip):
    delta = predict_ridge_delta(
        X=X,
        mean=mean,
        std=std,
        coef=coef,
        intercept=intercept,
        clip=clip
    )

    p_corr = P_BASE + lam * delta.T

    p_corr[0, :] = np.clip(p_corr[0, :], LOWER_BOUND[0], UPPER_BOUND[0])
    p_corr[1, :] = np.clip(p_corr[1, :], LOWER_BOUND[1], UPPER_BOUND[1])

    score_rmse = rmse(p_corr, p_true)
    score_mae = mae(p_corr, p_true)

    return score_rmse, score_mae, p_corr


# ============================================================
# 출력 포맷
# ============================================================

def print_array_for_main(name, arr):
    print(f"{name} = np.array(")
    print(repr(np.asarray(arr, dtype=float)))
    print(", dtype=float)")
    print()


# ============================================================
# main
# ============================================================

def main():
    p_train, d_train, p_val, d_val, p_bs = load_train_val_data()

    print("\n========== Data Shape ==========")
    print("p_train shape :", p_train.shape)
    print("d_train shape :", d_train.shape)
    print("p_val shape   :", p_val.shape)
    print("d_val shape   :", d_val.shape)
    print("p_bs shape    :", p_bs.shape)

    X_train, Y_train, P_BASE_train = build_feature_target(
        d_hat=d_train,
        p_true=p_train,
        p_bs=p_bs,
        name="train"
    )

    X_val, Y_val, P_BASE_val = build_feature_target(
        d_hat=d_val,
        p_true=p_val,
        p_bs=p_bs,
        name="val"
    )

    base_train_rmse = rmse(P_BASE_train, p_train)
    base_train_mae = mae(P_BASE_train, p_train)

    base_val_rmse = rmse(P_BASE_val, p_val)
    base_val_mae = mae(P_BASE_val, p_val)

    print("\n========== Base Algorithm Performance ==========")
    print(f"Train Base RMSE : {base_train_rmse:.4f} m")
    print(f"Train Base MAE  : {base_train_mae:.4f} m")
    print(f"Val Base RMSE   : {base_val_rmse:.4f} m")
    print(f"Val Base MAE    : {base_val_mae:.4f} m")

    # 너무 넓게 잡으면 train에는 좋아도 val/hidden에서 불안정할 수 있음
    alpha_list = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    lambda_list = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    clip_list = [2.0, 3.0, 5.0, 6.0, 8.0, 10.0]

    best = None

    print("\n========== Ridge Correction Search ==========")

    for alpha in alpha_list:
        mean, std, coef, intercept = fit_ridge_closed_form(
            X=X_train,
            Y=Y_train,
            alpha=alpha
        )

        for lam in lambda_list:
            for clip in clip_list:
                train_rmse, train_mae, _ = evaluate_with_correction(
                    P_BASE=P_BASE_train,
                    X=X_train,
                    p_true=p_train,
                    mean=mean,
                    std=std,
                    coef=coef,
                    intercept=intercept,
                    lam=lam,
                    clip=clip
                )

                val_rmse, val_mae, _ = evaluate_with_correction(
                    P_BASE=P_BASE_val,
                    X=X_val,
                    p_true=p_val,
                    mean=mean,
                    std=std,
                    coef=coef,
                    intercept=intercept,
                    lam=lam,
                    clip=clip
                )

                print(
                    f"alpha={alpha:7.2f}, lambda={lam:4.2f}, clip={clip:4.1f} "
                    f"-> Train RMSE={train_rmse:8.4f}, Val RMSE={val_rmse:8.4f}"
                )

                if best is None or val_rmse < best["val_rmse"]:
                    best = {
                        "alpha": alpha,
                        "lambda": lam,
                        "clip": clip,
                        "train_rmse": train_rmse,
                        "train_mae": train_mae,
                        "val_rmse": val_rmse,
                        "val_mae": val_mae,
                        "mean": mean,
                        "std": std,
                        "coef": coef,
                        "intercept": intercept,
                    }

    print("\n========== Best Ridge Setting ==========")
    print(f"BEST alpha  : {best['alpha']}")
    print(f"BEST lambda : {best['lambda']}")
    print(f"BEST clip   : {best['clip']}")
    print(f"Train RMSE  : {best['train_rmse']:.4f} m")
    print(f"Train MAE   : {best['train_mae']:.4f} m")
    print(f"Val RMSE    : {best['val_rmse']:.4f} m")
    print(f"Val MAE     : {best['val_mae']:.4f} m")

    print("\n========== Improvement ==========")
    print(f"Train RMSE improvement : {base_train_rmse - best['train_rmse']:.4f} m")
    print(f"Val RMSE improvement   : {base_val_rmse - best['val_rmse']:.4f} m")

    print("\n========== Constants for main.py ==========")
    print("아래 출력값을 main.py의 RIDGE_* 상수 부분에 복붙하면 됨.\n")

    np.set_printoptions(
        precision=8,
        suppress=True,
        linewidth=200
    )

    print(f"RIDGE_LAMBDA = {repr(float(best['lambda']))}")
    print(f"RIDGE_CLIP = {repr(float(best['clip']))}")
    print()

    print_array_for_main("RIDGE_MEAN", best["mean"])
    print_array_for_main("RIDGE_STD", best["std"])
    print_array_for_main("RIDGE_COEF", best["coef"])
    print_array_for_main("RIDGE_INTERCEPT", best["intercept"])

    # 결과 저장
    out_path = PROJECT_ROOT / "ridge_train_result.npz"
    np.savez(
        out_path,
        ridge_lambda=float(best["lambda"]),
        ridge_clip=float(best["clip"]),
        ridge_alpha=float(best["alpha"]),
        ridge_mean=best["mean"],
        ridge_std=best["std"],
        ridge_coef=best["coef"],
        ridge_intercept=best["intercept"],
        base_train_rmse=base_train_rmse,
        base_val_rmse=base_val_rmse,
        best_train_rmse=best["train_rmse"],
        best_val_rmse=best["val_rmse"],
    )

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
