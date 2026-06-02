# train.py
# Self-contained FINAL-FIT version
#
# 목적:
# 1) main.py에 들어가는 base 측위 알고리즘을 train.py 안에도 직접 포함한다.
# 2) 제공된 label 데이터 전체(보통 700개)를 사용해 anchor별 BIAS / SIGMA / BASE_WEIGHT를 다시 계산한다.
# 3) 전체 데이터에서 base 측위 결과 p_base의 남은 오차 p_true - p_base를 Ridge Regression으로 학습한다.
# 4) 전체 데이터 기준으로 alpha / lambda / clip을 탐색하고, main.py에 복붙할 상수를 출력한다.
#
# 주의:
# - 이 파일은 최종 제출 직전용 final-fit 코드이다.
# - 알고리즘 구조와 hyperparameter 후보는 이미 train/val/test 분할로 검증했다는 전제에서,
#   마지막에 공개된 700개 전체를 사용해 최종 상수를 재추정하는 용도이다.

import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares
from itertools import combinations
from pathlib import Path


EPS = 1e-9
MIN_SIGMA = 1.0


# ============================================================
# Base localization constants
# 아래 값은 fallback 기본값이다.
# main()에서 700개 전체 데이터로 BIAS / SIGMA / BASE_WEIGHT를 다시 계산해 덮어쓴다.
# ============================================================

BIAS = np.array(
    [10.996659, 10.096698, 7.935557, 8.307607, 12.560201, 11.877863,
     11.677825, 11.423726, 4.666315, 7.503596, 7.744365, 11.983844,
     11.174935, 10.835132, 10.476974, 7.755990, 8.293635, 13.739156],
    dtype=float
)

SIGMA = np.array(
    [16.044922, 15.089558, 12.504329, 12.741354, 18.052305, 16.970101,
     16.973554, 16.497124, 8.010103, 11.773311, 12.137595, 17.187890,
     16.288149, 15.844923, 15.350168, 11.738904, 12.688002, 19.495716],
    dtype=float
)

BASE_WEIGHT = np.array(
    [0.731475, 0.827031, 1.204353, 1.159961, 0.577842, 0.653892,
     0.653626, 0.691923, 2.934934, 1.358555, 1.278231, 0.637425,
     0.709792, 0.750057, 0.799187, 1.366531, 1.169737, 0.495446],
    dtype=float
)

TOP_K = 7
GATE_SCALE = 3.10
HUBER_DELTA = 11.50
SUBSET_SIZE = 4
MAX_CANDIDATES = 80
IRLS_ITER = 4

# 제공 데이터의 사용자 위치 범위를 기준으로 margin을 둔 clipping 범위
LOWER_BOUND = np.array([-70.0, -40.0], dtype=float)
UPPER_BOUND = np.array([70.0, 40.0], dtype=float)


# ============================================================
# Shape / data utility
# ============================================================

def ensure_2_by_n(arr, name="array"):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got shape {arr.shape}")
    if arr.shape[0] == 2:
        return arr
    if arr.shape[1] == 2:
        return arr.T
    raise ValueError(f"{name} must have shape (2, N) or (N, 2), got {arr.shape}")


def get_p_bs(data):
    if "p_bs" in data:
        return ensure_2_by_n(data["p_bs"], "p_bs")
    if "BS_positions" in data:
        return ensure_2_by_n(data["BS_positions"], "BS_positions")
    raise KeyError("p_bs 또는 BS_positions 변수를 찾을 수 없습니다.")


def get_p_true(data):
    if "p" not in data:
        raise KeyError("p 변수를 찾을 수 없습니다.")
    return ensure_2_by_n(data["p"], "p")


def get_d_hat(data):
    if "d_hat" not in data:
        raise KeyError("d_hat 변수를 찾을 수 없습니다.")
    d_hat = np.asarray(data["d_hat"], dtype=float)
    if d_hat.ndim != 2:
        raise ValueError(f"d_hat must be 2-dimensional, got shape {d_hat.shape}")
    return d_hat


def align_d_hat(d_hat, p_bs, p_true=None):
    """d_hat을 (num_anchor, num_user) 형태로 정렬한다."""
    d_hat = np.asarray(d_hat, dtype=float)
    p_bs = ensure_2_by_n(p_bs, "p_bs")

    num_anchor = p_bs.shape[1]

    if d_hat.shape[0] == num_anchor:
        out = d_hat
    elif d_hat.shape[1] == num_anchor:
        out = d_hat.T
    else:
        raise ValueError(
            f"d_hat shape {d_hat.shape} does not match num_anchor={num_anchor}"
        )

    if p_true is not None:
        p_true = ensure_2_by_n(p_true, "p_true")
        if out.shape[1] != p_true.shape[1]:
            raise ValueError(
                f"d_hat users {out.shape[1]} and p users {p_true.shape[1]} do not match"
            )

    return out


# ============================================================
# Evaluation functions
# ============================================================

def rmse(p_hat, p_true):
    p_hat = ensure_2_by_n(p_hat, "p_hat")
    p_true = ensure_2_by_n(p_true, "p_true")
    err = p_hat - p_true
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=0))))


def mae(p_hat, p_true):
    p_hat = ensure_2_by_n(p_hat, "p_hat")
    p_true = ensure_2_by_n(p_true, "p_true")
    err = p_hat - p_true
    return float(np.mean(np.sqrt(np.sum(err ** 2, axis=0))))


# ============================================================
# Calibration from all 700 labeled data
# ============================================================

def compute_calibration_constants(p_all, d_all, p_bs):
    """
    전체 공개 label 데이터로 anchor별 bias, sigma, reliability를 계산한다.

    bias_i  = median(d_hat_i - true_distance_i)
    sigma_i = std((d_hat_i - true_distance_i) - bias_i)
    weight_i = 1 / sigma_i^2 를 평균 1이 되도록 normalize
    """
    p_all = ensure_2_by_n(p_all, "p_all")
    p_bs = ensure_2_by_n(p_bs, "p_bs")
    d_all = align_d_hat(d_all, p_bs, p_all)

    use_m = min(d_all.shape[0], p_bs.shape[1])
    d_all = d_all[:use_m, :]
    p_bs = p_bs[:, :use_m]

    # true_dist: (num_anchor, num_user)
    diff = p_all[:, None, :] - p_bs[:, :, None]
    true_dist = np.sqrt(np.sum(diff ** 2, axis=0))

    err = d_all - true_dist

    bias = np.median(err, axis=1)
    centered = err - bias[:, None]
    sigma = np.std(centered, axis=1)
    sigma = np.maximum(sigma, MIN_SIGMA)

    raw_weight = 1.0 / (sigma ** 2 + EPS)
    base_weight = raw_weight / (np.mean(raw_weight) + EPS)

    return bias.astype(float), sigma.astype(float), base_weight.astype(float)


# ============================================================
# Robust weight / WLS base localization
# ============================================================

def huber_weight(residual, delta):
    residual = np.asarray(residual, dtype=float)
    abs_r = np.abs(residual)
    w = np.ones_like(abs_r)
    mask = abs_r > delta
    w[mask] = delta / (abs_r[mask] + EPS)
    return w


def wls_solve(d_cal, p_bs, weights, x0=None, max_nfev=60):
    """
    d_cal   : (M,)
    p_bs    : (2, M)
    weights : (M,)
    x0      : (2,)
    return  : (2,)
    """
    d_cal = np.asarray(d_cal, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    p_bs = ensure_2_by_n(p_bs, "p_bs")

    use_m = min(len(d_cal), len(weights), p_bs.shape[1])
    d_cal = d_cal[:use_m]
    weights = weights[:use_m]
    p_bs = p_bs[:, :use_m]

    valid = np.isfinite(d_cal) & np.isfinite(weights) & (weights > 0)

    if np.sum(valid) < 3:
        return np.mean(p_bs, axis=1).astype(float)

    anchors = p_bs[:, valid]
    d_use = d_cal[valid]
    w_use = weights[valid]

    if x0 is None:
        x0 = np.average(anchors, axis=1, weights=w_use)

    x0 = np.asarray(x0, dtype=float).reshape(2,)
    if not np.all(np.isfinite(x0)):
        x0 = np.mean(anchors, axis=1)

    def residual_func(x):
        x = np.asarray(x, dtype=float).reshape(2,)
        pred = np.sqrt(np.sum((anchors - x[:, None]) ** 2, axis=0))
        r = pred - d_use
        return np.sqrt(w_use) * r

    try:
        res = least_squares(
            residual_func,
            x0=x0,
            method="lm",
            max_nfev=max_nfev
        )
        x = np.asarray(res.x, dtype=float).reshape(2,)
        if np.all(np.isfinite(x)):
            return x
        return x0
    except Exception:
        return x0


def residual_cost(x, d_cal, p_bs, weights, huber_delta):
    """Huber loss 기반 후보 위치 residual cost."""
    x = np.asarray(x, dtype=float).reshape(2,)
    d_cal = np.asarray(d_cal, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    p_bs = ensure_2_by_n(p_bs, "p_bs")

    use_m = min(len(d_cal), len(weights), p_bs.shape[1])
    d_cal = d_cal[:use_m]
    weights = weights[:use_m]
    p_bs = p_bs[:, :use_m]

    pred = np.sqrt(np.sum((p_bs - x[:, None]) ** 2, axis=0))
    r = pred - d_cal
    abs_r = np.abs(r)

    loss = np.where(
        abs_r <= huber_delta,
        0.5 * r ** 2,
        huber_delta * (abs_r - 0.5 * huber_delta)
    )

    return float(np.sum(weights * loss))


def estimate_one_user(
    d_raw,
    p_bs,
    bias=None,
    sigma=None,
    base_weight=None,
    top_k=TOP_K,
    gate_scale=GATE_SCALE,
    huber_delta=HUBER_DELTA,
    subset_size=SUBSET_SIZE,
    max_candidates=MAX_CANDIDATES,
    irls_iter=IRLS_ITER
):
    """
    Base estimator:
    1. anchor별 median bias 보정
    2. anchor별 reliability 적용
    3. reliability 상위 TOP-K anchor 선택
    4. TOP-K anchor subset으로 여러 WLS 후보 생성
    5. 전체 anchor residual cost가 가장 작은 후보 선택
    6. residual gating
    7. Huber 기반 IRLS/WLS refinement
    """
    if bias is None:
        bias = BIAS
    if sigma is None:
        sigma = SIGMA
    if base_weight is None:
        base_weight = BASE_WEIGHT

    d_raw = np.asarray(d_raw, dtype=float).reshape(-1)
    p_bs = ensure_2_by_n(p_bs, "p_bs")

    use_m = min(len(d_raw), len(bias), len(sigma), len(base_weight), p_bs.shape[1])
    d_raw = d_raw[:use_m]
    p_bs = p_bs[:, :use_m]
    bias = np.asarray(bias, dtype=float)[:use_m]
    sigma = np.asarray(sigma, dtype=float)[:use_m]
    base_weight = np.asarray(base_weight, dtype=float)[:use_m]

    # 1. Bias 보정
    d_cal = d_raw - bias
    d_cal = np.maximum(d_cal, 0.1)

    # 2. 기본 anchor 신뢰도
    reliability = base_weight.copy()

    # 현재 사용자에서 측정값이 매우 큰 anchor는 약하게 감점
    if use_m > 0:
        bad_measure = d_raw > np.percentile(d_raw, 95)
        reliability[bad_measure] *= 0.5

    # 3. 신뢰도 상위 TOP-K anchor 선택
    top_k = min(top_k, use_m)
    good_idx = np.argsort(-reliability)[:top_k]

    # 4. anchor subset으로 여러 초기 후보 생성
    candidates = []
    subset_size = max(3, min(subset_size, len(good_idx)))
    comb_list = list(combinations(good_idx, subset_size))

    # 조합이 너무 많으면 reliability 합이 높은 조합만 사용
    if len(comb_list) > max_candidates:
        scored = []
        for comb in comb_list:
            score = np.sum(reliability[list(comb)])
            scored.append((score, comb))
        scored.sort(reverse=True, key=lambda x: x[0])
        comb_list = [c for _, c in scored[:max_candidates]]

    for comb in comb_list:
        comb = np.array(comb, dtype=int)
        w_subset = np.zeros(use_m, dtype=float)
        w_subset[comb] = reliability[comb]
        x0 = np.average(p_bs[:, comb], axis=1, weights=reliability[comb])

        x_candidate = wls_solve(
            d_cal=d_cal,
            p_bs=p_bs,
            weights=w_subset,
            x0=x0,
            max_nfev=60
        )

        if np.all(np.isfinite(x_candidate)):
            candidates.append(x_candidate)

    # 후보가 없으면 전체 anchor WLS로 대체
    if len(candidates) == 0:
        x0 = np.average(p_bs, axis=1, weights=reliability)
        candidates = [
            wls_solve(
                d_cal=d_cal,
                p_bs=p_bs,
                weights=reliability,
                x0=x0,
                max_nfev=60
            )
        ]

    # 5. 전체 anchor residual cost가 가장 작은 후보 선택
    costs = [
        residual_cost(
            x=c,
            d_cal=d_cal,
            p_bs=p_bs,
            weights=reliability,
            huber_delta=huber_delta
        )
        for c in candidates
    ]
    best_x = candidates[int(np.argmin(costs))]

    # 6. residual gating
    pred = np.sqrt(np.sum((p_bs - best_x[:, None]) ** 2, axis=0))
    residual = pred - d_cal

    gate_threshold = gate_scale * sigma
    keep = np.abs(residual) <= gate_threshold

    # 너무 많이 제거되면 residual이 작은 anchor를 최소 개수만큼 유지
    if np.sum(keep) < 4:
        keep = np.zeros(use_m, dtype=bool)
        keep_count = max(4, min(8, use_m))
        keep[np.argsort(np.abs(residual))[:keep_count]] = True

    # 7. Robust WLS / IRLS
    x = best_x.copy()
    final_weight = reliability.copy()
    final_weight[~keep] *= 0.05

    for _ in range(irls_iter):
        pred = np.sqrt(np.sum((p_bs - x[:, None]) ** 2, axis=0))
        residual = pred - d_cal

        h_w = huber_weight(residual, huber_delta)
        w = final_weight * h_w

        if np.sum(w > 0) < 3:
            break

        x_new = wls_solve(
            d_cal=d_cal,
            p_bs=p_bs,
            weights=w,
            x0=x,
            max_nfev=60
        )

        if not np.all(np.isfinite(x_new)):
            break

        if np.linalg.norm(x_new - x) < 1e-4:
            x = x_new
            break

        x = x_new

    x[0] = np.clip(x[0], LOWER_BOUND[0], UPPER_BOUND[0])
    x[1] = np.clip(x[1], LOWER_BOUND[1], UPPER_BOUND[1])

    return x.astype(float)


def your_algorithm(d_u, p_bs):
    """채점용 main.py의 your_algorithm과 같은 역할을 하는 base estimator."""
    return estimate_one_user(d_raw=d_u, p_bs=p_bs)


def run_algorithm(d_hat, p_bs):
    d_hat = np.asarray(d_hat, dtype=float)
    p_bs = ensure_2_by_n(p_bs, "p_bs")
    d_hat = align_d_hat(d_hat, p_bs)

    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user), dtype=float)

    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], p_bs)

    return p_hat


# ============================================================
# Ridge residual correction feature
# ============================================================

def make_ridge_feature(p_base, d_u, BS_positions):
    """
    Ridge residual correction용 feature 생성.

    핵심 아이디어:
    base estimator가 낸 위치 p_base에서 각 anchor까지의 예측거리와
    bias 보정된 측정거리 d_cal의 residual 패턴을 feature로 사용한다.
    Ridge는 이 residual 패턴으로부터 p_true - p_base를 학습한다.
    """
    p_base = np.asarray(p_base, dtype=float).reshape(2,)
    d_u = np.asarray(d_u, dtype=float).reshape(-1)
    p_bs = ensure_2_by_n(BS_positions, "BS_positions")

    use_m = min(len(d_u), len(BIAS), len(SIGMA), len(BASE_WEIGHT), p_bs.shape[1])
    d_u = d_u[:use_m]
    p_bs = p_bs[:, :use_m]
    bias = BIAS[:use_m]
    sigma = SIGMA[:use_m]
    base_weight = BASE_WEIGHT[:use_m]

    d_cal = np.maximum(d_u - bias, 0.1)
    pred = np.sqrt(np.sum((p_bs - p_base[:, None]) ** 2, axis=0))
    residual = pred - d_cal
    norm_residual = residual / (sigma + EPS)

    abs_norm = np.abs(norm_residual)
    clipped_norm = np.clip(norm_residual, -5.0, 5.0)

    # weighted statistics
    w = base_weight / (np.sum(base_weight) + EPS)
    weighted_mean = float(np.sum(w * residual))
    weighted_abs_mean = float(np.sum(w * np.abs(residual)))

    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)) + EPS)

    stats = np.array([
        p_base[0],
        p_base[1],
        float(np.mean(residual)),
        med,
        float(np.std(residual)),
        mad,
        float(np.mean(np.abs(residual))),
        weighted_mean,
        weighted_abs_mean,
        float(np.max(np.abs(residual))),
        float(np.mean(d_u)),
        float(np.std(d_u)),
    ], dtype=float)

    # anchor별 residual 패턴 + 요약 통계를 함께 사용
    feat = np.concatenate([
        stats,
        clipped_norm,
        abs_norm,
    ])

    return feat.astype(float)


# ============================================================
# Data loading for final full-data fit
# ============================================================

def unique_paths(paths):
    seen = set()
    out = []
    for p in paths:
        p = Path(p)
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def load_mat_file(path):
    data = sio.loadmat(path, squeeze_me=False)
    p = get_p_true(data)
    p_bs = get_p_bs(data)
    d_hat = align_d_hat(get_d_hat(data), p_bs, p)
    return p, d_hat, p_bs


def load_full_training_data():
    """
    전체 공개 label 데이터를 불러온다.

    우선순위:
    1) data_set/InF_DH_FR1.mat, data_set/DH_FR1.mat, 루트의 동일 파일
    2) full 파일이 없으면 train/val/test mat 파일을 모두 이어붙임
    """
    file_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidate_roots = unique_paths([
        cwd,
        file_dir,
        file_dir.parent,
    ])

    full_names = ["InF_DH_FR1.mat", "DH_FR1.mat"]
    full_candidates = []
    for root in candidate_roots:
        for name in full_names:
            full_candidates.append(root / "data_set" / name)
            full_candidates.append(root / name)

    full_candidates = unique_paths(full_candidates)

    for path in full_candidates:
        if path.exists():
            print("========== Load full training mat ==========")
            print("full:", path)
            p_all, d_all, p_bs = load_mat_file(path)
            return p_all, d_all, p_bs, str(path)

    split_names = ["train.mat", "val.mat", "validation.mat", "test.mat"]
    split_candidates = []
    for root in candidate_roots:
        for name in split_names:
            split_candidates.append(root / "data_set" / name)
            split_candidates.append(root / name)

    split_candidates = unique_paths(split_candidates)
    existing = [p for p in split_candidates if p.exists()]

    if not existing:
        raise FileNotFoundError(
            "InF_DH_FR1.mat / DH_FR1.mat 또는 train/val/test mat 파일을 찾을 수 없습니다."
        )

    print("========== Full mat 없음: split mat들을 이어붙임 ==========")
    for p in existing:
        print("split:", p)

    p_list = []
    d_list = []
    p_bs_ref = None

    for path in existing:
        try:
            p, d_hat, p_bs = load_mat_file(path)
        except KeyError:
            # p가 없는 hidden/test 형태 파일이면 학습에 사용할 수 없으므로 skip
            print("skip unlabeled file:", path)
            continue

        if p_bs_ref is None:
            p_bs_ref = p_bs
        else:
            if p_bs.shape != p_bs_ref.shape or not np.allclose(p_bs, p_bs_ref):
                raise ValueError(f"p_bs mismatch in {path}")

        p_list.append(p)
        d_list.append(d_hat)

    if not p_list:
        raise FileNotFoundError("label p가 들어 있는 mat 파일을 찾을 수 없습니다.")

    p_all = np.concatenate(p_list, axis=1)
    d_all = np.concatenate(d_list, axis=1)

    return p_all, d_all, p_bs_ref, "split-concat"


# ============================================================
# Ridge closed-form learning
# ============================================================

def fit_ridge_closed_form(X, Y, alpha):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    mean = X.mean(axis=0)
    std = X.std(axis=0) + EPS
    Xn = (X - mean) / std

    X_aug = np.hstack([
        Xn,
        np.ones((Xn.shape[0], 1), dtype=float)
    ])

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
# Feature / target build
# ============================================================

def build_feature_target(d_hat, p_true, p_bs, name="dataset"):
    p_true = ensure_2_by_n(p_true, "p_true")
    p_bs = ensure_2_by_n(p_bs, "p_bs")
    d_hat = align_d_hat(d_hat, p_bs, p_true)

    num_user = d_hat.shape[1]
    X = []
    Y = []
    P_BASE = np.zeros((2, num_user), dtype=float)

    print(f"\n========== Build Features: {name} ==========")

    for u in range(num_user):
        d_u = d_hat[:, u]
        p_base = your_algorithm(d_u, p_bs)
        feat = make_ridge_feature(
            p_base=p_base,
            d_u=d_u,
            BS_positions=p_bs
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
# Printing utilities
# ============================================================

def print_array_for_main(name, arr):
    arr_text = np.array2string(
        np.asarray(arr, dtype=float),
        precision=8,
        separator=", ",
        suppress_small=False,
        max_line_width=200
    )
    print(f"{name} = np.array(")
    print(arr_text)
    print(", dtype=float)")
    print()


# ============================================================
# train.py main: full 700 final fit
# ============================================================

def main():
    global BIAS, SIGMA, BASE_WEIGHT

    project_root = Path(__file__).resolve().parent

    p_all, d_all, p_bs, source_name = load_full_training_data()

    print("\n========== Full Data Shape ==========")
    print("source     :", source_name)
    print("p_all shape:", p_all.shape)
    print("d_all shape:", d_all.shape)
    print("p_bs shape :", p_bs.shape)

    # 1) 전체 700개로 anchor별 calibration 상수 재계산
    BIAS, SIGMA, BASE_WEIGHT = compute_calibration_constants(
        p_all=p_all,
        d_all=d_all,
        p_bs=p_bs
    )

    print("\n========== Full-data Calibration Constants ==========")
    print("BIAS shape       :", BIAS.shape)
    print("SIGMA shape      :", SIGMA.shape)
    print("BASE_WEIGHT shape:", BASE_WEIGHT.shape)

    # 2) 전체 700개로 base estimator 실행 후 Ridge feature/target 생성
    X_all, Y_all, P_BASE_all = build_feature_target(
        d_hat=d_all,
        p_true=p_all,
        p_bs=p_bs,
        name="full_700"
    )

    base_all_rmse = rmse(P_BASE_all, p_all)
    base_all_mae = mae(P_BASE_all, p_all)

    print("\n========== Base Algorithm Full-data Performance ==========")
    print(f"Full Base RMSE : {base_all_rmse:.4f} m")
    print(f"Full Base MAE  : {base_all_mae:.4f} m")

    # 3) 전체 700개 기준으로 Ridge alpha/lambda/clip 탐색
    # 이미 validation으로 정한 best 값이 있다면 아래 리스트를 하나만 남겨도 된다.
    alpha_list = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    lambda_list = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    clip_list = [2.0, 3.0, 5.0, 6.0, 8.0, 10.0]

    best = None

    print("\n========== Ridge Correction Search on Full 700 ==========")

    for alpha in alpha_list:
        mean, std, coef, intercept = fit_ridge_closed_form(
            X=X_all,
            Y=Y_all,
            alpha=alpha
        )

        for lam in lambda_list:
            for clip in clip_list:
                full_rmse, full_mae, _ = evaluate_with_correction(
                    P_BASE=P_BASE_all,
                    X=X_all,
                    p_true=p_all,
                    mean=mean,
                    std=std,
                    coef=coef,
                    intercept=intercept,
                    lam=lam,
                    clip=clip
                )

                print(
                    f"alpha={alpha:7.2f}, lambda={lam:4.2f}, clip={clip:4.1f} "
                    f"-> Full RMSE={full_rmse:8.4f}, Full MAE={full_mae:8.4f}"
                )

                if best is None or full_rmse < best["full_rmse"]:
                    best = {
                        "alpha": alpha,
                        "lambda": lam,
                        "clip": clip,
                        "full_rmse": full_rmse,
                        "full_mae": full_mae,
                        "mean": mean,
                        "std": std,
                        "coef": coef,
                        "intercept": intercept,
                    }

    print("\n========== Best Full-data Ridge Setting ==========")
    print(f"BEST alpha  : {best['alpha']}")
    print(f"BEST lambda : {best['lambda']}")
    print(f"BEST clip   : {best['clip']}")
    print(f"Full RMSE   : {best['full_rmse']:.4f} m")
    print(f"Full MAE    : {best['full_mae']:.4f} m")

    print("\n========== Improvement on Full 700 ==========")
    print(f"Full RMSE improvement : {base_all_rmse - best['full_rmse']:.4f} m")
    print(f"Full MAE improvement  : {base_all_mae - best['full_mae']:.4f} m")

    print("\n========== Constants for main.py ==========")
    print("아래 출력값을 main.py의 상수 부분에 복붙하면 됨.\n")

    np.set_printoptions(
        precision=8,
        suppress=True,
        linewidth=200
    )

    print_array_for_main("BIAS", BIAS)
    print_array_for_main("SIGMA", SIGMA)
    print_array_for_main("BASE_WEIGHT", BASE_WEIGHT)

    print(f"TOP_K = {TOP_K}")
    print(f"GATE_SCALE = {repr(float(GATE_SCALE))}")
    print(f"HUBER_DELTA = {repr(float(HUBER_DELTA))}")
    print(f"SUBSET_SIZE = {SUBSET_SIZE}")
    print(f"MAX_CANDIDATES = {MAX_CANDIDATES}")
    print(f"IRLS_ITER = {IRLS_ITER}")
    print()

    print("USE_RIDGE_CORRECTION = True")
    print(f"RIDGE_LAMBDA = {repr(float(best['lambda']))}")
    print(f"RIDGE_CLIP = {repr(float(best['clip']))}")
    print()

    print_array_for_main("RIDGE_MEAN", best["mean"])
    print_array_for_main("RIDGE_STD", best["std"])
    print_array_for_main("RIDGE_COEF", best["coef"])
    print_array_for_main("RIDGE_INTERCEPT", best["intercept"])

    out_path = project_root / "model_ridge.npz"
    np.savez(
        out_path,
        bias=BIAS,
        sigma=SIGMA,
        base_weight=BASE_WEIGHT,
        top_k=int(TOP_K),
        gate_scale=float(GATE_SCALE),
        huber_delta=float(HUBER_DELTA),
        subset_size=int(SUBSET_SIZE),
        max_candidates=int(MAX_CANDIDATES),
        irls_iter=int(IRLS_ITER),
        ridge_lambda=float(best["lambda"]),
        ridge_clip=float(best["clip"]),
        ridge_alpha=float(best["alpha"]),
        ridge_mean=best["mean"],
        ridge_std=best["std"],
        ridge_coef=best["coef"],
        ridge_intercept=best["intercept"],
        base_full_rmse=base_all_rmse,
        base_full_mae=base_all_mae,
        best_full_rmse=best["full_rmse"],
        best_full_mae=best["full_mae"],
    )

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
