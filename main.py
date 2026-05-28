import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares
from itertools import combinations


EPS = 1e-9


# ============================================================
# 위치 범위 제한
# ============================================================

LOWER_BOUND = np.array([-75.0, -45.0], dtype=float)
UPPER_BOUND = np.array([75.0, 45.0], dtype=float)


# ============================================================
# 기존 Robust WLS 상수
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
    [0.879054, 0.934709, 1.127957, 1.106974, 0.781304, 0.831129,
     0.830960, 0.854958, 1.760819, 1.197993, 1.162038, 0.820598,
     0.865927, 0.890149, 0.918840, 1.201504, 1.111629, 0.723459],
    dtype=float
)

TOP_K = 8
GATE_SCALE = 3.5
HUBER_DELTA = 12.0
TUKEY_SCALE = 6.0


# ============================================================
# Ridge residual correction 상수
# BEST alpha  : 10.0  -> main.py에서는 직접 사용하지 않음
# BEST lambda : 0.7
# BEST clip   : 10.0
# ============================================================

USE_RIDGE_CORRECTION = True

RIDGE_LAMBDA = 0.7
RIDGE_CLIP = 10.0

RIDGE_MEAN = np.array(
    [-1.00363402, 0.20381142, -4.9609419, 0.04348387,
     17.99137933, -53.44466856, 16.1583436, 13.11373269,
     8.88844666, 13.37996642, 53.6492451, 37.36296058,
     2.11813762, 54.23574344, 51.7272261, 31.50360934,
     6.72344254, 119.91263747, 9.21, 8.79],
    dtype=float
)

RIDGE_STD = np.array(
    [36.88344401, 18.28069415, 4.2887264, 3.29753158,
     6.25144295, 28.1220502, 5.711448, 3.58394183,
     2.68962846, 6.47582137, 27.94037331, 14.96267512,
     1.79015783, 10.9663433, 13.58488186, 7.43188553,
     5.34335726, 30.42616089, 1.60682918, 1.60682918],
    dtype=float
)

RIDGE_COEF = np.array(
    [[-1.47568727,  0.13128036],
     [-0.26799764, -3.70255228],
     [-1.48293027,  1.62018295],
     [-0.47703594, -0.39829799],
     [-0.29305686, -2.48739713],
     [ 0.13133212,  4.01562553],
     [ 0.32642163, -0.93174208],
     [-1.22244507, -0.81935823],
     [ 0.38753715,  0.81855700],
     [-0.06607256,  2.43574805],
     [-0.07499726,  4.11858544],
     [ 0.15851597,  1.73771431],
     [ 0.48009736,  0.34722768],
     [-0.94875971, -0.94450266],
     [-0.10054269,  1.36171790],
     [ 1.05545577,  0.42119122],
     [ 0.00216388,  0.65326895],
     [-0.11212374, -0.81884271],
     [ 0.34698478,  0.04694043],
     [-0.34698478, -0.04694043]],
    dtype=float
)

RIDGE_INTERCEPT = np.array(
    [-0.06427757, -0.28628681],
    dtype=float
)


# ============================================================
# 평가 함수
# ============================================================

def rmse(p_hat, p_true):
    err = p_hat - p_true
    return float(np.sqrt(np.mean(np.sum(err ** 2, axis=0))))


def mae(p_hat, p_true):
    err = p_hat - p_true
    return float(np.mean(np.sqrt(np.sum(err ** 2, axis=0))))


# ============================================================
# Huber weight
# ============================================================

def huber_weight(residual, delta):
    residual = np.asarray(residual, dtype=float)
    abs_r = np.abs(residual)

    w = np.ones_like(abs_r)

    mask = abs_r > delta
    w[mask] = delta / (abs_r[mask] + EPS)

    return w


# ============================================================
# Tukey weight
# ============================================================

def tukey_weight(residual, sigma, tukey_scale=TUKEY_SCALE):
    residual = np.asarray(residual, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    sigma = np.maximum(sigma, EPS)

    u = residual / (tukey_scale * sigma)

    w = np.zeros_like(u)

    mask = np.abs(u) < 1.0
    w[mask] = (1.0 - u[mask] ** 2) ** 2

    return w


# ============================================================
# Bounded WLS solver
# ============================================================

def wls_solve(d_cal, BS_positions, weights, x0=None, max_nfev=60):
    d_cal = np.asarray(d_cal, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    BS_positions = np.asarray(BS_positions, dtype=float)

    valid = np.isfinite(d_cal) & np.isfinite(weights) & (weights > 0)

    if np.sum(valid) < 3:
        x_fallback = np.mean(BS_positions, axis=1).astype(float)
        x_fallback = np.minimum(np.maximum(x_fallback, LOWER_BOUND), UPPER_BOUND)
        return x_fallback

    anchors = BS_positions[:, valid]
    d_use = d_cal[valid]
    w_use = weights[valid]

    if x0 is None:
        x0 = np.average(anchors, axis=1, weights=w_use)

    x0 = np.asarray(x0, dtype=float).reshape(2,)

    if not np.all(np.isfinite(x0)):
        x0 = np.mean(anchors, axis=1)

    x0 = np.minimum(np.maximum(x0, LOWER_BOUND), UPPER_BOUND)

    def residual_func(x):
        x = np.asarray(x, dtype=float).reshape(2,)
        pred = np.sqrt(np.sum((anchors - x[:, None]) ** 2, axis=0))
        r = pred - d_use
        return np.sqrt(w_use) * r

    try:
        res = least_squares(
            residual_func,
            x0=x0,
            method="trf",
            bounds=(LOWER_BOUND, UPPER_BOUND),
            max_nfev=max_nfev
        )

        x = np.asarray(res.x, dtype=float).reshape(2,)

        if np.all(np.isfinite(x)):
            x = np.minimum(np.maximum(x, LOWER_BOUND), UPPER_BOUND)
            return x

        return x0

    except Exception:
        return x0


# ============================================================
# 후보 위치 cost
# sigma 정규화 포함
# ============================================================

def residual_cost(x, d_cal, BS_positions, weights, sigma, huber_delta):
    x = np.asarray(x, dtype=float).reshape(2,)
    d_cal = np.asarray(d_cal, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    sigma = np.asarray(sigma, dtype=float).reshape(-1)

    pred = np.sqrt(np.sum((BS_positions - x[:, None]) ** 2, axis=0))

    r_raw = pred - d_cal
    r = r_raw / (sigma + EPS)

    abs_r = np.abs(r)

    loss = np.where(
        abs_r <= huber_delta,
        0.5 * r ** 2,
        huber_delta * (abs_r - 0.5 * huber_delta)
    )

    return float(np.sum(weights * loss))


# ============================================================
# 사용자 1명 위치 추정
# ============================================================

def your_algorithm(d_u, BS_positions):
    d_u = np.asarray(d_u, dtype=float).reshape(-1)
    BS_positions = np.asarray(BS_positions, dtype=float)

    use_m = min(d_u.shape[0], BS_positions.shape[1], BIAS.shape[0])

    d_u = d_u[:use_m]
    BS_positions = BS_positions[:, :use_m]

    bias = BIAS[:use_m]
    sigma = SIGMA[:use_m]
    base_weight = BASE_WEIGHT[:use_m]

    # --------------------------------------------------------
    # 1. Bias 보정
    # --------------------------------------------------------
    d_cal = d_u - bias
    d_cal = np.maximum(d_cal, 0.1)

    # --------------------------------------------------------
    # 2. Anchor 기본 신뢰도
    # --------------------------------------------------------
    reliability = base_weight.copy()

    bad_measure = d_cal > (np.median(d_cal) + 2.5 * np.median(sigma))
    reliability[bad_measure] *= 0.3

    # --------------------------------------------------------
    # 3. 신뢰도 상위 TOP_K anchor 선택
    # --------------------------------------------------------
    top_k = min(TOP_K, use_m)
    good_idx = np.argsort(-reliability)[:top_k]

    # --------------------------------------------------------
    # 4. Anchor subset으로 여러 초기 후보 생성
    # --------------------------------------------------------
    subset_size = max(3, min(4, len(good_idx)))
    comb_list = list(combinations(good_idx, subset_size))

    max_candidates = 80

    if len(comb_list) > max_candidates:
        scored = []

        for comb in comb_list:
            score = np.sum(reliability[list(comb)])
            scored.append((score, comb))

        scored.sort(reverse=True, key=lambda x: x[0])
        comb_list = [c for _, c in scored[:max_candidates]]

    candidates = []

    for comb in comb_list:
        comb = np.array(comb, dtype=int)

        w_subset = np.zeros(use_m, dtype=float)
        w_subset[comb] = reliability[comb]

        x0 = np.average(
            BS_positions[:, comb],
            axis=1,
            weights=reliability[comb]
        )

        x0 = np.minimum(np.maximum(x0, LOWER_BOUND), UPPER_BOUND)

        x_candidate = wls_solve(
            d_cal=d_cal,
            BS_positions=BS_positions,
            weights=w_subset,
            x0=x0,
            max_nfev=60
        )

        if np.all(np.isfinite(x_candidate)):
            candidates.append(x_candidate)

    if len(candidates) == 0:
        x0 = np.average(BS_positions, axis=1, weights=reliability)
        x0 = np.minimum(np.maximum(x0, LOWER_BOUND), UPPER_BOUND)

        candidates = [
            wls_solve(
                d_cal=d_cal,
                BS_positions=BS_positions,
                weights=reliability,
                x0=x0,
                max_nfev=60
            )
        ]

    # --------------------------------------------------------
    # 5. residual cost 최소 후보 선택
    # --------------------------------------------------------
    costs = [
        residual_cost(
            x=c,
            d_cal=d_cal,
            BS_positions=BS_positions,
            weights=reliability,
            sigma=sigma,
            huber_delta=HUBER_DELTA
        )
        for c in candidates
    ]

    x = candidates[int(np.argmin(costs))]
    x = np.minimum(np.maximum(x, LOWER_BOUND), UPPER_BOUND)

    # --------------------------------------------------------
    # 6. Residual gating
    # --------------------------------------------------------
    pred = np.sqrt(np.sum((BS_positions - x[:, None]) ** 2, axis=0))
    residual = pred - d_cal

    gate_threshold = GATE_SCALE * sigma
    keep = np.abs(residual) <= gate_threshold

    if np.sum(keep) < 4:
        keep = np.zeros(use_m, dtype=bool)
        keep_count = max(4, min(8, use_m))
        keep[np.argsort(np.abs(residual))[:keep_count]] = True

    final_weight = reliability.copy()
    final_weight[~keep] *= 0.02

    # --------------------------------------------------------
    # 7. Tukey + Huber Robust IRLS/WLS
    # --------------------------------------------------------
    for _ in range(4):
        pred = np.sqrt(np.sum((BS_positions - x[:, None]) ** 2, axis=0))
        residual = pred - d_cal

        t_w = tukey_weight(
            residual=residual,
            sigma=sigma,
            tukey_scale=TUKEY_SCALE
        )

        h_w = huber_weight(
            residual=residual,
            delta=HUBER_DELTA
        )

        w = final_weight * t_w * h_w

        if np.sum(w > 1e-6) < 3:
            break

        x_new = wls_solve(
            d_cal=d_cal,
            BS_positions=BS_positions,
            weights=w,
            x0=x,
            max_nfev=60
        )

        if not np.all(np.isfinite(x_new)):
            break

        x_new = np.minimum(np.maximum(x_new, LOWER_BOUND), UPPER_BOUND)

        if np.linalg.norm(x_new - x) < 1e-4:
            x = x_new
            break

        x = x_new

    x = np.minimum(np.maximum(x, LOWER_BOUND), UPPER_BOUND)

    # --------------------------------------------------------
    # 8. Ridge residual correction
    # --------------------------------------------------------
    x = apply_ridge_correction(
        p_base=x,
        d_u=d_u,
        BS_positions=BS_positions
    )

    return x.astype(float)


# ============================================================
# Ridge feature 생성
# ============================================================

def make_ridge_feature(p_base, d_u, BS_positions):
    p_base = np.asarray(p_base, dtype=float).reshape(2,)
    d_u = np.asarray(d_u, dtype=float).reshape(-1)
    BS_positions = np.asarray(BS_positions, dtype=float)

    use_m = min(d_u.shape[0], BS_positions.shape[1], BIAS.shape[0])

    d_u = d_u[:use_m]
    BS_positions = BS_positions[:, :use_m]

    bias = BIAS[:use_m]
    sigma = SIGMA[:use_m]

    d_cal = d_u - bias
    d_cal = np.maximum(d_cal, 0.1)

    pred = np.sqrt(np.sum((BS_positions - p_base.reshape(2, 1)) ** 2, axis=0))

    residual = pred - d_cal
    abs_res = np.abs(residual)

    sorted_abs = np.sort(abs_res)
    if len(sorted_abs) >= 3:
        top3_abs_mean = np.mean(sorted_abs[-3:])
    else:
        top3_abs_mean = np.mean(sorted_abs)

    norm_res = residual / (sigma + EPS)
    robust_cost = np.mean(np.minimum(norm_res ** 2, HUBER_DELTA ** 2))

    pos_count = np.sum(residual > 0)
    neg_count = np.sum(residual < 0)

    feat = np.array([
        p_base[0],
        p_base[1],

        np.mean(residual),
        np.median(residual),
        np.std(residual),
        np.min(residual),
        np.max(residual),

        np.mean(abs_res),
        np.median(abs_res),
        np.std(abs_res),
        np.max(abs_res),
        top3_abs_mean,

        robust_cost,

        np.mean(d_cal),
        np.median(d_cal),
        np.std(d_cal),
        np.min(d_cal),
        np.max(d_cal),

        pos_count,
        neg_count,
    ], dtype=float)

    return feat


# ============================================================
# Ridge residual correction 적용
# ============================================================

def apply_ridge_correction(p_base, d_u, BS_positions):
    p_base = np.asarray(p_base, dtype=float).reshape(2,)

    if not USE_RIDGE_CORRECTION:
        return p_base.astype(float)

    feat = make_ridge_feature(
        p_base=p_base,
        d_u=d_u,
        BS_positions=BS_positions
    )

    feat_n = (feat - RIDGE_MEAN) / (RIDGE_STD + EPS)

    delta = feat_n @ RIDGE_COEF + RIDGE_INTERCEPT
    delta = np.asarray(delta, dtype=float).reshape(2,)

    delta = np.clip(delta, -RIDGE_CLIP, RIDGE_CLIP)

    p_final = p_base + RIDGE_LAMBDA * delta

    p_final = np.minimum(np.maximum(p_final, LOWER_BOUND), UPPER_BOUND)

    return p_final.astype(float)


# ============================================================
# main: 채점 양식 맞춤
# ============================================================

def main():
    # 1) 입력 데이터 로드 — 채점기가 같은 폴더에 .mat 파일 자동 배치
    mat_path = "DH_FR1.mat"

    data = sio.loadmat(mat_path, squeeze_me=False)

    if "BS_positions" in data:
        BS_positions = np.asarray(data["BS_positions"], dtype=float)
    elif "p_bs" in data:
        BS_positions = np.asarray(data["p_bs"], dtype=float)
    else:
        raise KeyError("BS_positions 또는 p_bs 변수를 찾을 수 없습니다.")

    d_hat = np.asarray(data["d_hat"], dtype=float)

    # p는 채점 결과 반환에는 필요 없지만, 로컬 확인용으로만 사용 가능
    p = None
    if "p" in data:
        p = np.asarray(data["p"], dtype=float)

    # 2) 본인 알고리즘 — 사용자 수는 입력에서 동적으로 받기
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user), dtype=float)

    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], BS_positions)

    # 로컬에서 p가 있을 때만 성능 확인
    if p is not None:
        print("========== Performance ==========")
        print("p shape    :", p.shape)
        print("d_hat shape:", d_hat.shape)
        print("BS shape   :", BS_positions.shape)
        print(f"RMSE       : {rmse(p_hat, p):.4f} m")
        print(f"MAE        : {mae(p_hat, p):.4f} m")
        print("p_hat shape:", p_hat.shape)

    # 3) 결과 반환 — numpy 배열, 모양 (2, num_user)
    return p_hat


if __name__ == "__main__":
    main()