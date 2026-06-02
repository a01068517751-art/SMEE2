# main.py
# Bias-Calibrated Anchor Subset Guided Robust WLS
# + Ridge Residual Correction
#
# 실행:
# python main.py
#
# 입력 파일:
# DH_FR1.mat
#
# 출력:
# p_hat: (2, num_user)

import os
import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares
from itertools import combinations


EPS = 1e-9


# ============================================================
# train.py에서 700개 전체 데이터로 다시 학습한 상수
# ============================================================

BIAS = np.array(
    [11.74162945, 10.17816517, 8.27765462, 7.97363333, 11.67672767, 12.56919529,
     11.84382724, 9.96984110, 6.36890169, 7.75988061, 8.94242160, 11.57797246,
     11.15832581, 10.83513230, 9.73967005, 7.96321859, 8.75623895, 13.14241766],
    dtype=float
)

SIGMA = np.array(
    [25.38524115, 18.59119098, 18.34508854, 19.80495307, 19.54973525, 20.70912136,
     20.65342779, 19.52827431, 20.38547654, 18.79188004, 20.23261748, 22.54638482,
     17.81067701, 19.49466307, 20.40355171, 18.27389921, 20.48438760, 20.28179907],
    dtype=float
)

BASE_WEIGHT = np.array(
    [0.61358574, 1.14399314, 1.17489274, 1.00806888, 1.03456095, 0.92196513,
     0.92694414, 1.03683610, 0.95147221, 1.11968892, 0.96590341, 0.77782879,
     1.24645610, 1.04041445, 0.94978717, 1.18406459, 0.94230582, 0.96122463],
    dtype=float
)

TOP_K = 7
GATE_SCALE = 3.10
HUBER_DELTA = 11.50
SUBSET_SIZE = 4
MAX_CANDIDATES = 80
IRLS_ITER = 4

LOWER_BOUND = np.array([-70.0, -40.0], dtype=float)
UPPER_BOUND = np.array([70.0, 40.0], dtype=float)


# ============================================================
# Ridge residual correction constants
# ============================================================

USE_RIDGE_CORRECTION = True
RIDGE_LAMBDA = 0.7
RIDGE_CLIP = 10.0

RIDGE_MEAN = np.array(
    [-2.29219659, -0.18650746, -4.03398058, 0.90852285, 17.96608337, 8.55302680,
     13.06651188, -3.95401503, 12.93495642, 52.78164583, 63.83176763, 31.95984781,
     -0.22613085, -0.17721464, -0.20349745, -0.23346070, -0.15037449, -0.16876632,
     -0.18219254, -0.18316130, -0.25296156, -0.17494311, -0.21127892, -0.22040178,
     -0.20842929, -0.17553476, -0.16292263, -0.19293935, -0.22851788, -0.16881988,
     0.62503000, 0.65453939, 0.65408196, 0.65668414, 0.65170769, 0.62982435,
     0.60753056, 0.67445554, 0.64920739, 0.65804037, 0.68181195, 0.62803382,
     0.65861645, 0.68299971, 0.65656996, 0.63615217, 0.65357551, 0.67278147],
    dtype=float
)

RIDGE_STD = np.array(
    [37.54959937, 18.55722409, 3.77453042, 3.05536279, 6.32176919, 2.97992542,
     3.47275236, 3.64910318, 3.38749956, 28.49581789, 10.96292568, 7.34249128,
     0.93702274, 0.93042870, 0.93844616, 0.93358582, 0.89667295, 0.90159185,
     0.88402818, 0.93825443, 0.97500019, 0.91523976, 0.98834123, 0.92382618,
     0.91213053, 0.97117124, 0.92665271, 0.91565955, 0.95691576, 0.93128701,
     0.80710771, 0.69479688, 0.72465356, 0.75228218, 0.68620419, 0.74752362,
     0.72357875, 0.71797689, 0.80895696, 0.71567525, 0.76900618, 0.76062181,
     0.67775264, 0.72253954, 0.75235571, 0.71037385, 0.73628864, 0.69559818],
    dtype=float
)

RIDGE_COEF = np.array(
    [[-3.63709816e+00, -1.31672940e-01],
     [-2.04659503e-01, -3.53460401e+00],
     [-1.06891752e+00, 1.11600360e+01],
     [-6.07682500e-01, -2.29747705e-01],
     [1.77339765e+00, 4.11739123e-01],
     [-7.66664754e-01, -2.50241051e-01],
     [2.76550076e-02, 3.25977398e-01],
     [1.69033635e+00, -1.22960212e+01],
     [5.96902398e-02, 2.74875845e-04],
     [-1.46934851e+00, -1.64162561e-02],
     [-1.67927722e-01, 6.79169310e-01],
     [2.06164509e-01, -8.03968965e-01],
     [-1.54953366e+00, 2.06329919e+00],
     [-3.67030054e-01, 3.76088879e+00],
     [-1.37889863e-01, 3.50779658e+00],
     [7.22223996e-01, 2.41913021e+00],
     [5.10385296e-01, 2.90118841e+00],
     [1.39712843e+00, 1.64420974e+00],
     [-1.13566950e+00, 2.50921205e-01],
     [-1.30975180e+00, 4.62007399e-01],
     [-1.03006711e+00, 1.26059856e+00],
     [9.87930723e-02, 1.01611691e+00],
     [1.74184142e+00, 3.39711692e-01],
     [1.09548548e+00, 5.87490938e-01],
     [-1.32127721e+00, -7.22938187e-02],
     [-3.45408637e-01, -2.11050397e+00],
     [-4.68766633e-01, -1.97784988e+00],
     [8.87876356e-02, -8.82494978e-01],
     [9.48628380e-01, -3.03020066e+00],
     [6.48588459e-01, -1.68350482e+00],
     [-1.23655677e+00, 2.56880783e+00],
     [-1.55371325e-01, 2.29511258e+00],
     [1.60787567e-01, 1.78349143e+00],
     [8.19205682e-01, 1.26424189e+00],
     [4.84616873e-01, 2.21587803e+00],
     [1.02320766e+00, 9.63510196e-01],
     [-1.26687695e+00, -9.20975141e-02],
     [-9.00070267e-01, -7.31018897e-01],
     [-6.90187715e-01, 7.26487166e-01],
     [1.67674161e-01, 4.32667236e-01],
     [9.80067225e-01, 3.23548333e-01],
     [1.06580928e+00, 9.03030359e-01],
     [-1.18446366e+00, -1.12830330e+00],
     [-4.68407162e-01, -2.31391755e+00],
     [-4.15899223e-01, -2.17707189e+00],
     [2.03993499e-01, -2.16671206e+00],
     [1.01992119e+00, -3.00094885e+00],
     [6.22974511e-01, -1.61333673e+00]],
    dtype=float
)

RIDGE_INTERCEPT = np.array(
    [-0.03489419, -0.26875556],
    dtype=float
)


# ============================================================
# Utility
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
# Robust weight
# ============================================================

def huber_weight(residual, delta):
    residual = np.asarray(residual, dtype=float)
    abs_r = np.abs(residual)
    w = np.ones_like(abs_r)

    mask = abs_r > delta
    w[mask] = delta / (abs_r[mask] + EPS)

    return w


# ============================================================
# WLS 위치 추정
# ============================================================

def wls_solve(d_cal, p_bs, weights, x0=None, max_nfev=60):
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


# ============================================================
# 후보 위치 residual cost 계산
# ============================================================

def residual_cost(x, d_cal, p_bs, weights, huber_delta):
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


# ============================================================
# Base 위치 추정
# ============================================================

def estimate_one_user(
    d_raw,
    p_bs,
    bias=BIAS,
    sigma=SIGMA,
    base_weight=BASE_WEIGHT,
    top_k=TOP_K,
    gate_scale=GATE_SCALE,
    huber_delta=HUBER_DELTA,
    subset_size=SUBSET_SIZE,
    max_candidates=MAX_CANDIDATES,
    irls_iter=IRLS_ITER
):
    d_raw = np.asarray(d_raw, dtype=float).reshape(-1)
    p_bs = ensure_2_by_n(p_bs, "p_bs")

    use_m = min(len(d_raw), len(bias), len(sigma), len(base_weight), p_bs.shape[1])
    d_raw = d_raw[:use_m]
    p_bs = p_bs[:, :use_m]
    bias = np.asarray(bias, dtype=float)[:use_m]
    sigma = np.asarray(sigma, dtype=float)[:use_m]
    base_weight = np.asarray(base_weight, dtype=float)[:use_m]

    d_cal = d_raw - bias
    d_cal = np.maximum(d_cal, 0.1)

    reliability = base_weight.copy()

    if use_m > 0:
        bad_measure = d_raw > np.percentile(d_raw, 95)
        reliability[bad_measure] *= 0.5

    top_k = min(top_k, use_m)
    good_idx = np.argsort(-reliability)[:top_k]

    candidates = []

    subset_size = max(3, min(subset_size, len(good_idx)))
    comb_list = list(combinations(good_idx, subset_size))

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

    pred = np.sqrt(np.sum((p_bs - best_x[:, None]) ** 2, axis=0))
    residual = pred - d_cal

    gate_threshold = gate_scale * sigma
    keep = np.abs(residual) <= gate_threshold

    if np.sum(keep) < 4:
        keep = np.zeros(use_m, dtype=bool)
        keep_count = max(4, min(8, use_m))
        keep[np.argsort(np.abs(residual))[:keep_count]] = True

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


# ============================================================
# Ridge residual correction
# ============================================================

def make_ridge_feature(p_base, d_u, p_bs):
    p_base = np.asarray(p_base, dtype=float).reshape(2,)
    d_u = np.asarray(d_u, dtype=float).reshape(-1)
    p_bs = ensure_2_by_n(p_bs, "p_bs")

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

    feat = np.concatenate([
        stats,
        clipped_norm,
        abs_norm,
    ])

    return feat.astype(float)


def predict_ridge_delta_from_feature(feat):
    feat = np.asarray(feat, dtype=float).reshape(1, -1)

    Xn = (feat - RIDGE_MEAN) / (RIDGE_STD + EPS)

    delta = Xn @ RIDGE_COEF + RIDGE_INTERCEPT
    delta = np.asarray(delta, dtype=float).reshape(2,)

    delta = np.clip(delta, -RIDGE_CLIP, RIDGE_CLIP)

    return delta


def apply_ridge_correction(p_base, d_u, p_bs):
    if not USE_RIDGE_CORRECTION:
        return p_base.astype(float)

    feat = make_ridge_feature(
        p_base=p_base,
        d_u=d_u,
        p_bs=p_bs
    )

    delta = predict_ridge_delta_from_feature(feat)

    p_corr = p_base + RIDGE_LAMBDA * delta

    p_corr[0] = np.clip(p_corr[0], LOWER_BOUND[0], UPPER_BOUND[0])
    p_corr[1] = np.clip(p_corr[1], LOWER_BOUND[1], UPPER_BOUND[1])

    return p_corr.astype(float)


# ============================================================
# 채점용 함수
# ============================================================

def your_algorithm(d_u, p_bs):
    p_base = estimate_one_user(
        d_raw=d_u,
        p_bs=p_bs
    )

    p_final = apply_ridge_correction(
        p_base=p_base,
        d_u=d_u,
        p_bs=p_bs
    )

    return p_final.astype(float)


def run_algorithm(d_hat, p_bs):
    d_hat = np.asarray(d_hat, dtype=float)
    p_bs = ensure_2_by_n(p_bs, "p_bs")

    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user), dtype=float)

    for u in range(num_user):
        p_hat[:, u] = your_algorithm(
            d_u=d_hat[:, u],
            p_bs=p_bs
        )

    return p_hat


# ============================================================
# main
# ============================================================

def main():
    mat_path = "DH_FR1.mat"

    if not os.path.exists(mat_path):
        raise FileNotFoundError("DH_FR1.mat file was not found.")

    data = sio.loadmat(mat_path, squeeze_me=False)

    p_bs = get_p_bs(data)
    d_hat = np.asarray(data["d_hat"], dtype=float)

    p_hat = run_algorithm(
        d_hat=d_hat,
        p_bs=p_bs
    )

    if "p" in data:
        p_true = ensure_2_by_n(data["p"], "p")
        if p_true.shape == p_hat.shape:
            print("========== Local Performance ==========")
            print(f"RMSE : {rmse(p_hat, p_true):.4f} m")
            print(f"MAE  : {mae(p_hat, p_true):.4f} m")

    print("p_hat shape:", p_hat.shape)

    return p_hat


if __name__ == "__main__":
    main()
