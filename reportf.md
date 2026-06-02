# Indoor Localization Using Anchor Residual-Aware Robust WLS with Bounded Ridge Compensation
- 학번: 12223626
- 이름: 김우주

## 1. Motivation & Introduction

실내 RTT 기반 측위에서는 anchor와 사용자 사이의 거리 측정값이 실제 거리와 크게 달라지는 경우가 많다. 제공된 데이터셋에서도 RTT distance는 전반적으로 실제 거리보다 크게 측정되는 positive bias를 보였고, 일부 anchor에서는 매우 큰 거리 오차가 발생하는 heavy-tailed noise 특성이 나타났다. 이러한 특성 때문에 모든 anchor를 동일한 신뢰도로 사용하는 단순 least squares 또는 weighted least squares 방식은 이상치 anchor에 쉽게 흔들릴 수 있다.

중간 발표까지 사용하였던 기존 구조는 WiFi-guide 기반 UWB gating 방식이었다. 먼저 WiFi distance를 이용해 guide 위치를 추정하고, 그 위치를 기준으로 UWB anchor를 선별한 뒤, 이후 IRLS 기반 WLS로 위치를 refinement하는 흐름이었다. 이 구조는 guide 위치가 정확할 때는 효과가 있지만, guide가 부정확하면 이후 gating도 잘못된 기준을 따라간다는 한계가 있었다. 즉, 정상 anchor가 제거되거나 이상치 anchor가 유지될 수 있었고, 이는 최종 위치 추정 오차로 이어질 수 있었다.

최종 알고리즘에서는 이 문제를 해결하기 위해 WiFi guide와 같은 외부 초기 위치를 먼저 신뢰하지 않는 구조로 변경하였다. 대신 anchor distance 자체의 residual consistency를 이용해 여러 위치 후보를 만들고, 그중 전체 anchor와 가장 일관적인 후보를 초기 위치로 선택하였다. 이후 residual gating과 Huber 기반 IRLS/WLS를 적용하여 outlier anchor의 영향을 줄였다.

Robust WLS 이후에도 일부 샘플에서는 특정 방향으로 반복적인 잔여 오차가 남았다. 이는 단순 random noise라기보다 anchor 배치, distance bias, residual pattern에 의해 발생하는 systematic residual error로 판단하였다. 따라서 최종 단계에서는 물리 기반 추정 결과를 완전히 대체하지 않고, 해당 결과가 남긴 residual pattern을 이용해 제한적으로 dx, dy만 보정하는 bounded Ridge residual compensation을 추가하였다.

최종 알고리즘은 다음 흐름으로 구성된다.

| Step | Description |
|---|---|
| 1 | Anchor-wise bias calibration |
| 2 | Anchor reliability estimation |
| 3 | TOP-K anchor selection |
| 4 | Anchor subset candidate generation |
| 5 | Residual-based candidate selection |
| 6 | Residual gating |
| 7 | Huber 기반 robust IRLS/WLS refinement |
| 8 | Bounded Ridge residual compensation |

이 구조의 핵심은 Ridge regression으로 위치를 직접 예측하는 것이 아니다. 먼저 anchor geometry와 residual consistency를 이용한 robust localization pipeline으로 기본 위치 p_base를 계산하고, 그 위치에서 계산되는 residual feature를 이용해 남은 systematic error만 제한적으로 보정한다. 따라서 본 알고리즘은 direct black-box machine learning localization이 아니라, physics-guided robust localization에 lightweight residual learning을 결합한 hybrid 구조이다.

## 2. Algorithm Description

### 2.1 Dataset Analysis

제공된 데이터는 사용자 실제 위치 p, RTT 기반 거리 측정값 d_hat, 그리고 anchor 좌표 p_bs 또는 BS_positions로 구성되어 있다. 한 명의 사용자에 대해 여러 anchor의 거리 측정값이 주어지며, 최종 목표는 각 사용자에 대해 2차원 위치 p_hat을 추정하는 것이다.

거리 측정값을 실제 거리와 비교한 결과 다음과 같은 특징이 관찰되었다.

| Property | Observation |
|---|---|
| Positive bias | 측정 거리가 실제 거리보다 전반적으로 크게 나타남 |
| Heavy-tailed noise | 일부 anchor distance가 매우 크게 튀는 long-tail 오차 발생 |
| Anchor-wise reliability difference | anchor마다 오차 scale과 안정성이 다르게 나타남 |
| No missing value | NaN 및 Inf가 없어 결측치 대체보다 거리 오차 보정이 중요함 |
| Hidden test 존재 | 제공 데이터에만 맞춘 암기형 모델은 일반화 성능이 낮을 수 있음 |

따라서 단순히 모든 anchor를 같은 비중으로 사용하는 방식보다는, anchor별 bias와 reliability를 반영하고, residual이 큰 anchor의 영향을 줄이는 robust estimation 구조가 필요하다고 판단하였다.

알고리즘 개발 과정에서는 제공된 700개 데이터를 train, validation, test를 500/100/100개의 구조로 나누어 알고리즘의 타당성을 체크하였다. 이 단계에서 TOP-K anchor 수, residual gate scale, Huber threshold, Ridge alpha, Ridge lambda, Ridge clip 등을 비교하였다. 최종 제출용 모델에서는 선택된 구조와 hyperparameter 후보를 고정한 뒤, 제공된 label 데이터 전체 700개를 사용하여 BIAS, SIGMA, BASE_WEIGHT, Ridge coefficient를 다시 추정하였다. 이는 hidden test label을 사용하지 않는 범위에서 가능한 전체 공개 데이터를 활용한 final-fit 과정이다.

### 2.2 Anchor-wise Bias Calibration

RTT 기반 거리 측정값은 실내 환경에서 실제 거리보다 크게 측정되는 경향이 있었다. 이를 완화하기 위해 anchor별 bias를 계산하였다.

사용자 u와 anchor i에 대해 실제 거리는 다음과 같다.

| Symbol | Definition |
|---|---|
| d_true,i(u) | sqrt((x_u - x_i)^2 + (y_u - y_i)^2) |
| e_i(u) | d_hat,i(u) - d_true,i(u) |

Anchor i의 bias는 평균 대신 median으로 계산하였다.

| Quantity | Definition |
|---|---|
| b_i | median(e_i(u)) |
| d_cal,i | d_i - b_i |

Median을 사용한 이유는 일부 anchor에서 매우 큰 outlier가 발생하기 때문이다. Mean은 extreme outlier에 쉽게 흔들리지만, median은 long-tail noise에 상대적으로 강하다. 또한 bias 보정 후 거리가 비정상적으로 작아지는 경우를 막기 위해 최소 거리 하한을 두었다.

최종 제출용 모델에서는 제공된 700개 전체 label 데이터를 이용해 anchor별 bias를 다시 계산하였다. 최종 BIAS는 다음과 같다.

| Anchor | BIAS |
|---:|---:|
| 0 | 11.7416 |
| 1 | 10.1782 |
| 2 | 8.2777 |
| 3 | 7.9736 |
| 4 | 11.6767 |
| 5 | 12.5692 |
| 6 | 11.8438 |
| 7 | 9.9698 |
| 8 | 6.3689 |
| 9 | 7.7599 |
| 10 | 8.9424 |
| 11 | 11.5780 |
| 12 | 11.1583 |
| 13 | 10.8351 |
| 14 | 9.7397 |
| 15 | 7.9632 |
| 16 | 8.7562 |
| 17 | 13.1424 |

### 2.3 Anchor Reliability Estimation

Anchor마다 measurement stability가 다르므로, anchor별 residual scale을 계산하였다. Residual scale이 큰 anchor는 신뢰도가 낮다고 판단하고 weight를 낮추었으며, residual scale이 작은 anchor는 상대적으로 높은 weight를 부여하였다.

| Quantity | Definition |
|---|---|
| sigma_i | std((d_hat,i - d_true,i) - b_i) |
| w_i | 1 / (sigma_i^2 + epsilon) |

실제 구현에서는 weight가 지나치게 한쪽으로 치우치지 않도록 평균이 1 근처가 되도록 normalize한 BASE_WEIGHT를 사용하였다. 또한 특정 사용자 샘플에서 측정값이 전체 거리 분포에 비해 지나치게 큰 경우, 해당 anchor는 sample-level outlier일 가능성이 있으므로 reliability를 추가로 감소시켰다.

최종 제출용 모델에서 사용한 SIGMA와 BASE_WEIGHT는 다음과 같다.

| Anchor | SIGMA | BASE_WEIGHT |
|---:|---:|---:|
| 0 | 25.3852 | 0.6136 |
| 1 | 18.5912 | 1.1440 |
| 2 | 18.3451 | 1.1749 |
| 3 | 19.8050 | 1.0081 |
| 4 | 19.5497 | 1.0346 |
| 5 | 20.7091 | 0.9220 |
| 6 | 20.6534 | 0.9269 |
| 7 | 19.5283 | 1.0368 |
| 8 | 20.3855 | 0.9515 |
| 9 | 18.7919 | 1.1197 |
| 10 | 20.2326 | 0.9659 |
| 11 | 22.5464 | 0.7778 |
| 12 | 17.8107 | 1.2465 |
| 13 | 19.4947 | 1.0404 |
| 14 | 20.4036 | 0.9498 |
| 15 | 18.2739 | 1.1841 |
| 16 | 20.4844 | 0.9423 |
| 17 | 20.2818 | 0.9612 |

### 2.4 TOP-K Anchor Selection

모든 anchor를 초기 위치 생성에 그대로 사용하는 대신, reliability가 높은 anchor를 우선적으로 선택하였다. Residual variance가 큰 anchor는 실제 위치와의 consistency가 낮을 가능성이 높기 때문에, 초기 후보 생성 단계에서는 reliability 기준 상위 anchor를 중심으로 사용하였다.

| Parameter | Selected Value | Role |
|---|---:|---|
| TOP_K | 7 | 초기 후보 생성에 사용할 신뢰도 상위 anchor 개수 |

이 단계는 unstable anchor가 초기 위치 후보를 왜곡하는 것을 줄이는 역할을 한다. 다만 최종 residual cost와 refinement에서는 전체 anchor consistency를 다시 평가하므로, 단순히 일부 anchor만 영구적으로 사용하는 방식은 아니다.

### 2.5 Anchor Subset Candidate Generation

선택된 TOP-K anchor들에 대해 여러 subset 조합을 생성하였다. 각 subset은 독립적인 위치 후보를 만들며, weighted least squares를 이용하여 해당 subset에 가장 잘 맞는 위치를 계산하였다.

| Parameter | Selected Value | Role |
|---|---:|---|
| SUBSET_SIZE | 4 | 하나의 후보 위치를 만들 때 사용하는 anchor 개수 |
| MAX_CANDIDATES | 80 | 후보 조합 수 제한 |

Subset S_k에 대한 위치 후보는 다음 목적함수를 최소화하는 위치로 정의된다.

| Quantity | Definition |
|---|---|
| p_k | argmin_p Σ_{i in S_k} w_i (||p - a_i|| - d_cal,i)^2 |

각 subset은 서로 다른 anchor 조합을 사용하므로, 특정 anchor configuration에 과도하게 의존하지 않고 여러 가능한 위치 후보를 만들 수 있다. Candidate 수가 지나치게 많아지면 runtime이 증가하므로, 후보 조합이 많을 경우 reliability 합이 높은 subset을 우선적으로 사용하였다.

### 2.6 Residual-based Candidate Selection

생성된 여러 위치 후보 중에서 전체 anchor distance와 가장 일관성이 높은 후보를 초기 위치로 선택하였다.

Candidate p_k에 대한 residual은 다음과 같이 계산된다.

| Quantity | Definition |
|---|---|
| r_i(p_k) | ||p_k - a_i|| - d_cal,i |
| J(p_k) | Σ_i w_i rho(r_i(p_k)) |

여기서 rho는 Huber loss이다. Huber loss를 사용한 이유는 작은 residual에는 제곱 오차처럼 반응하면서도, 큰 residual에는 영향이 선형적으로 제한되기 때문이다. 최종 초기 위치는 candidate cost J가 가장 작은 후보로 선택된다.

이 방식은 WiFi guide처럼 외부 초기 위치를 직접 신뢰하지 않고, anchor residual consistency 자체로 초기 위치를 선택한다는 점에서 기존 방식과 다르다.

### 2.7 Residual Gating

선택된 초기 위치를 기준으로 residual gating을 수행하였다. Anchor residual이 지나치게 큰 경우 해당 anchor는 이상치일 가능성이 높다고 판단하였다.

| Quantity | Definition or Value |
|---|---|
| tau_i | GATE_SCALE × sigma_i |
| GATE_SCALE | 3.10 |
| Gate condition | |r_i| <= tau_i |

Gate를 통과하지 못한 anchor는 완전히 제거하지 않고 weight를 매우 작게 낮추었다. 완전 제거 대신 soft down-weighting을 사용한 이유는 hidden test에서 특정 anchor가 우연히 큰 residual을 보이더라도, 해당 anchor의 정보를 완전히 잃는 것을 방지하기 위해서이다. 만약 gate를 통과한 anchor 수가 너무 적으면 WLS가 불안정해질 수 있으므로, residual이 작은 anchor를 최소 개수 이상 유지하도록 하였다.

### 2.8 Huber Robust IRLS/WLS Refinement

Residual gating 이후에는 IRLS 기반 robust weighted least squares를 반복 수행하였다. 이 단계에서는 residual이 큰 anchor의 영향을 반복적으로 줄이며 위치를 refine한다.

Huber weight는 다음과 같이 계산된다.

| Condition | Huber Weight |
|---|---|
| |r_i| <= delta | 1 |
| |r_i| > delta | delta / |r_i| |

최종 weight는 base reliability, residual gating 결과, Huber weight를 함께 반영한다. 이후 WLS를 반복 수행하여 위치를 갱신한다.

| Parameter | Selected Value | Role |
|---|---:|---|
| HUBER_DELTA | 11.50 | Huber loss 및 Huber weight threshold |
| IRLS_ITER | 4 | Robust WLS 반복 횟수 |

최종 base 위치는 다음 목적함수를 최소화하는 형태로 계산된다.

| Quantity | Definition |
|---|---|
| p_base | argmin_p Σ_i w_final,i (||p - a_i|| - d_cal,i)^2 |

반복은 위치 변화량이 충분히 작아지거나 최대 반복 횟수에 도달하면 종료하였다. 이 결과 p_base는 물리 기반 robust localization pipeline이 산출한 기본 위치이다.

### 2.9 Bounded Ridge Residual Compensation

Robust WLS 결과만으로도 outlier 영향을 줄일 수 있었지만, 전체 데이터와 validation 결과를 분석하면 일부 샘플에서 residual pattern에 따른 systematic error가 남아 있었다. 따라서 최종 단계에서는 Ridge regression을 이용하여 p_base의 남은 위치 오차 dx, dy를 보정하였다.

중요한 점은 Ridge regression이 d_hat에서 위치를 직접 예측하지 않는다는 것이다. Ridge는 p_base와 p_base에서 계산한 residual feature를 입력으로 받아, p_base가 실제 위치에서 얼마나 벗어났는지만 학습한다.

Ridge feature에는 다음 정보가 포함된다.

| Feature Group | Meaning |
|---|---|
| p_base_x, p_base_y | robust WLS가 산출한 기본 위치 |
| residual mean, median, std | p_base 기준 anchor residual의 전체 경향 |
| MAD 및 absolute residual statistics | residual 분산과 outlier 정도 |
| weighted residual statistics | anchor reliability를 반영한 residual 경향 |
| max absolute residual | 가장 큰 anchor inconsistency 정도 |
| raw distance statistics | 입력 거리값의 전체 scale |
| normalized anchor residual pattern | anchor별 residual 방향과 크기 |
| absolute normalized residual pattern | anchor별 residual magnitude |

Target은 다음과 같이 정의하였다.

| Quantity | Definition |
|---|---|
| delta_true | p_true - p_base |
| delta_ridge | X_norm W + b |

Ridge regression은 다음 objective를 최소화한다.

| Objective | Meaning |
|---|---|
| min_W ||Y - XW||^2 + alpha ||W||^2 | coefficient가 과도하게 커지는 것을 억제하면서 residual correction 학습 |

최종 위치는 Ridge 보정값을 그대로 적용하지 않고, clip과 lambda mixture를 적용하였다.

| Quantity | Definition |
|---|---|
| delta_clipped | clip(delta_ridge, -C, C) |
| p_final | p_base + lambda × delta_clipped |

이 bounded correction 구조를 사용한 이유는 hidden test에서 Ridge 보정이 과하게 튀는 것을 방지하기 위해서이다. 즉, 최종 위치는 물리 기반 p_base를 중심으로 제한된 범위 안에서만 수정된다.

### 2.10 Final Hyperparameters and Model File

최종 제출용 모델에서는 제공된 700개 전체 데이터를 사용해 calibration 상수와 Ridge coefficient를 다시 추정하였다. train.py 실행 결과 Ridge 관련 학습 결과는 model_ridge.npz에도 저장하였다. 다만 최종 main.py는 실행 안정성을 위해 model_ridge.npz를 직접 로드하지 않고, 학습된 값을 상수로 코드 안에 포함한다.
이는 알고리즘의 견고성을 확인하기 위해 모든 알고리즘이 개발된 후, 모든 데이터를 넣어 다시 파라미터들을 골랐다.

최종 선택된 parameter는 다음과 같다.

| Parameter | Selected Value | Role |
|---|---:|---|
| TOP_K | 7 | 초기 후보 생성에 사용할 신뢰도 상위 anchor 개수 |
| GATE_SCALE | 3.10 | residual gating threshold scale |
| HUBER_DELTA | 11.50 | Huber loss 및 Huber weight threshold |
| SUBSET_SIZE | 4 | subset candidate generation에 사용할 anchor 수 |
| MAX_CANDIDATES | 80 | 최대 후보 조합 수 |
| IRLS_ITER | 4 | robust WLS 반복 횟수 |
| Ridge alpha | 0.10 | Ridge coefficient regularization strength |
| Ridge lambda | 0.70 | Ridge 보정값을 최종 위치에 반영하는 비율 |
| Ridge clip | 10.00 | Ridge 보정량의 최대 허용 범위 |

## 2.11 Relationship to References and Original Contribution

이번 알고리즘 구현에는 robust estimation, Ridge regression, UWB NLOS error mitigation 관련 기존 연구를 참고하였다. 참고문헌은 각 단계의 이론적 배경과 관련 방법을 이해하기 위해 사용하였고, 실제 최종 알고리즘 구조는 제공 데이터의 residual 특성과 validation 결과를 바탕으로 직접 설계하였다.

Huber [1]는 outlier가 포함된 데이터에서 큰 residual의 영향을 제한하는 robust estimation 개념을 제안하였다. 이 개념을 anchor distance residual에 적용하여, Huber loss에서 유도되는 residual weight를 반복적인 WLS refinement에 사용하였다. 다만 Huber 논문의 location parameter estimation 문제를 그대로 푼 것이 아니라, 실내 측위의 nonlinear anchor-distance residual 문제에 맞게 변형하여 사용하였다.

Hoerl and Kennard [2]는 Ridge regression의 정규화 개념을 제안하였다. 이번 알고리즘에는 Ridge를 위치 직접 예측 모델로 사용하지 않고, robust WLS가 산출한 p_base의 남은 오차 p_true - p_base를 보정하는 residual correction에만 사용하였다. 또한 hidden test에서 보정값이 과도하게 커지는 것을 막기 위해 lambda와 clip을 적용한 bounded Ridge compensation 구조로 제한하였다.

Wang et al. [3]은 UWB indoor positioning에서 NLOS propagation이 주요한 positioning/ranging error 원인이라는 점과, residual weighting 및 WLS 계열의 NLOS error mitigation 방법들을 정리하였다. 이 내용을 바탕으로 anchor-wise median bias calibration, residual scale estimation, reliability weighting, residual gating을 사용하였다. 그러나 각 anchor를 LOS/NLOS로 직접 분류하지 않고, residual consistency와 soft weighting으로 신뢰도가 낮은 측정값의 영향을 줄이는 방식을 사용하였다.

Fan and Du [4]는 WLS 기반으로 신뢰도가 낮은 UWB measurement의 영향을 줄일 수 있다는 관련 사례로 참고하였다. 하지만 해당 연구의 핵심인 Kalman filter와 Mahalanobis distance 기반 NLOS identification은 본 프로젝트에 직접 적용하지 않았다. 이번 실험 환경은 시간 순서가 있는 tracking 문제가 아니라 각 사용자 샘플이 독립적인 static localization 문제로 처리되므로, 본 알고리즘은 Kalman filtering 대신 anchor subset candidate generation, 전체 anchor residual consistency 기반 후보 선택, soft residual gating, Huber 기반 IRLS/WLS를 사용하였다.

## 3. Agent AI Usage

이번 실험에서는 ChatGPT를 포함한 Agent AI를 알고리즘을 대신 설계하는 도구가 아니라, 데이터 분석과 실험 결과를 해석하고 정리하는 보조 도구로 활용하였다. 알고리즘의 핵심 방향 결정, parameter 선택, 성능 비교, 최종 구조 채택 여부는 직접 실험한 RMSE/MAE 결과와 데이터 특성을 기준으로 판단하였다.

먼저 제공된 RTT 데이터의 특성을 직접 확인한 뒤, d_hat이 실제 거리보다 전반적으로 크게 측정되는 positive bias를 가진다는 점과 일부 anchor에서 큰 outlier가 발생한다는 점을 분석하였다. 이 과정에서 AI는 이러한 현상이 왜 일반 WLS 성능을 떨어뜨리는지, median bias 보정이나 residual scale 기반 weighting이 어떤 의미를 가지는지 이론적으로 정리하는 데 보조적으로 활용되었다.

중간 발표 이후에는 기존 WiFi-guide 기반 UWB gating 방식의 한계를 다시 검토하였다. WiFi guide를 유지할지, grid search 방식으로 바꿀지, anchor residual consistency 기반으로 초기 위치를 만들지 여러 방향을 비교하였다. 이때 AI는 각 후보 방식의 장단점, 계산량 증가 가능성, hidden test에서의 일반화 위험을 정리하는 보조 역할을 하였다.

알고리즘 개발 과정에서는 여러 후보를 직접 구현하고 성능을 확인하였다. 초기에는 bias calibration, anchor reliability weighting, coarse/fine grid search, residual gating, Huber 기반 WLS, RANSAC-like subset candidate 방식 등을 단계적으로 검토하였다. AI는 각 방식이 어떤 오차 요인에 대응하는지 설명하거나, 실험 결과가 예상과 다르게 나왔을 때 가능한 원인을 정리하는 데 사용하였다. 그러나 실제로 어떤 방식을 폐기하고 어떤 방식을 유지할지는 validation 성능, 실행 시간, 구조의 안정성을 기준으로 직접 결정하였다.

최종적으로는 계산량이 큰 grid search 방식보다, reliability가 높은 anchor subset에서 여러 후보 위치를 만들고 전체 anchor residual cost로 후보를 선택하는 구조가 더 적합하다고 판단하였다. 이는 실험을 통해 성능과 실행 시간을 비교한 뒤 직접 선택한 것이다. 

Ridge residual compensation을 추가하는 과정에서도 먼저 robust WLS 이후 남는 오차 양상을 확인하였다. 일부 샘플에서 p_base가 특정 방향으로 반복적으로 벗어나는 경향이 있어, 이를 residual pattern 기반의 systematic error로 해석하였다. AI는 Ridge regression을 위치 직접 예측 모델로 쓰는 경우와, p_base의 residual correction만 학습하는 경우의 차이를 정리하는 데 도움을 주었다. 최종적으로는 direct ML localization이 아니라, robust WLS 결과를 기준으로 제한된 dx, dy만 보정하는 bounded Ridge compensation 구조를 선택하였다.

또한 AI는 train.py와 main.py의 역할을 구분하는 데에도 보조적으로 활용되었다. train.py에는 anchor bias, sigma, base weight, Ridge coefficient를 학습하는 과정을 포함하고, main.py에는 학습된 상수를 이용한 추론 과정만 남기도록 구조를 정리하였다.


## 4. Result & Discussion

### 4.1 Evaluation Method

위치 추정 성능은 RMSE와 MAE를 기준으로 평가하였다.

| Metric | Definition |
|---|---|
| RMSE | sqrt((1/N) Σ_i ||p_i - p_hat,i||^2) |
| MAE | (1/N) Σ_i ||p_i - p_hat,i|| |

알고리즘 개발 단계에서는 제공된 700개 데이터를 train, validation, test로 나누어 구조와 parameter를 검토하였다. 이 과정에서 guide 기반 구조, grid search 기반 구조, anchor subset 기반 구조, robust WLS, Ridge residual compensation의 효과를 비교하였다. 최종 제출용 모델에서는 선택된 구조를 바탕으로 제공된 700개 전체 label 데이터를 사용해 calibration 상수와 Ridge coefficient를 다시 추정하였다.

Hidden test set에는 정답 p가 공개되지 않으므로, 최종 main.py는 hidden test에서 추가 학습을 수행하지 않는다. main.py는 d_hat과 p_bs만 입력으로 받아 p_hat을 반환하며, 이미 train.py에서 학습된 상수를 이용해 base robust WLS와 Ridge residual compensation을 적용한다.

### 4.2 Quantitative Result

최종 제출용 final-fit 과정에서 제공된 700개 전체 label 데이터 기준 성능은 다음과 같다.

| Model | RMSE | MAE |
|---|---:|---:|
| Robust WLS base | 10.8010 | 8.5485 |
| Robust WLS + bounded Ridge compensation | 8.6487 | 6.7815 |

Ridge compensation을 추가했을 때 전체 데이터 기준 개선량은 다음과 같다.

| Metric | Base | Final | Improvement |
|---|---:|---:|---:|
| RMSE | 10.8010 | 8.6487 | 2.1523 |
| MAE | 8.5485 | 6.7815 | 1.7670 |

이 결과는 hidden test 성능이 아니라, 제공된 700개 전체 label 데이터를 사용한 final-fit 성능이다. 따라서 이 수치만으로 hidden test 성능을 보장할 수는 없다. 다만 알고리즘 개발 단계에서 train/validation/test 분할을 통해 구조를 먼저 검토한 뒤, 최종 제출 직전에 전체 공개 데이터를 사용해 상수를 재추정했다는 점에서, hidden label을 사용하지 않는 합리적인 final training 과정이라고 판단하였다.

### 4.3 Baseline Comparison and Development Process

이번 실험은 단순 WLS에서 시작하여 단계적으로 robust estimation 구조를 추가하면서 발전하였다. 초기에는 모든 anchor를 동일한 신뢰도로 사용하는 basic WLS를 고려하였다. 하지만 RTT distance에는 positive bias와 heavy-tailed noise가 존재하므로, 일부 이상치 anchor가 위치 추정 결과를 크게 왜곡하였다.

다음 단계에서는 anchor-wise bias calibration을 적용하였다. 각 anchor의 median error를 계산하고 이를 측정 거리에서 제거함으로써, RTT distance가 실제 거리보다 지속적으로 크게 측정되는 현상을 완화하고자 하였다.

이후 anchor reliability weighting을 추가하였다. Anchor마다 residual scale이 다르게 나타났기 때문에, variance가 큰 anchor의 weight를 낮추고 안정적인 anchor의 weight를 높였다. 이를 통해 unstable anchor가 위치 추정에 미치는 영향을 줄일 수 있었다.

처음에는 coarse/fine grid search 기반 global search도 고려하였다. 전체 공간에서 residual consistency가 가장 좋은 위치를 탐색하는 방식은 guide dependency를 줄인다는 장점이 있었지만, 계산량이 증가하고 validation 성능 개선이 제한적이었다. 이에 따라 최종적으로는 reliability가 높은 anchor subset을 여러 개 만들고, 각 subset에서 생성된 후보 중 전체 anchor residual cost가 가장 작은 후보를 선택하는 anchor subset consensus 방식을 사용하였다.

마지막으로 robust WLS 결과에서 남는 systematic residual을 줄이기 위해 bounded Ridge residual compensation을 추가하였다. 이 단계는 위치를 직접 예측하는 ML 모델이 아니라, 기존 robust WLS 결과 p_base의 residual feature를 이용해 남은 dx, dy만 제한적으로 보정하는 구조이다.

전체 개발 과정은 다음과 같이 정리할 수 있다.

| Stage | Main Idea | Reason for Adoption or Rejection |
|---|---|---|
| Basic WLS | 모든 anchor 동일 사용 | outlier에 취약하여 한계 존재 |
| Bias calibration | anchor별 median bias 보정 | RTT positive bias 완화에 필요 |
| Reliability weighting | anchor별 residual scale 기반 weight | unstable anchor 영향 감소 |
| WiFi-guide fusion | guide 위치 기준 gating | guide가 부정확하면 gating도 흔들림 |
| Grid-based search | 전체 공간 residual consistency 탐색 | 계산량 증가 및 성능 개선 제한 |
| Anchor subset consensus | 여러 anchor subset 후보 생성 후 residual cost로 선택 | guide 의존성 감소 및 robust initialization 가능 |
| Huber IRLS/WLS | residual 기반 반복 reweighting | heavy-tailed noise에 대응 |
| Bounded Ridge compensation | p_base의 residual pattern으로 dx, dy 제한 보정 | robust WLS 이후 남는 systematic error 감소 |

이 과정은 단순히 RMSE 숫자만 줄이는 방향이 아니라, hidden environment에서도 일반화 가능한 구조를 찾는 과정이었다. 특히 Ridge compensation을 마지막에 추가하였지만, 전체 알고리즘의 중심은 여전히 anchor geometry와 residual consistency를 이용한 물리 기반 robust estimation이다.

### 4.4 Discussion About the Proposed Method

이 알고리즘은 특정 guide 위치 dependency를 줄이고, anchor residual consistency 자체를 기반으로 위치를 추정하도록 설계하였다. 이 방향은 기존 WiFi-guide 방식의 한계를 보완하는 데 적절했다고 판단한다.

첫째, anchor-wise bias calibration은 RTT distance의 지속적인 positive bias를 완화하는 데 도움이 되었다. 실제 거리보다 크게 측정되는 경향을 anchor별 median bias로 제거함으로써, 이후 WLS가 더 합리적인 거리값을 사용하도록 만들었다.

둘째, anchor reliability weighting은 unstable anchor의 영향을 줄이는 데 효과적이었다. 모든 anchor를 동일하게 사용하는 것보다 residual scale 기반 weighting이 더 안정적인 결과를 제공하였다.

셋째, anchor subset consensus 구조는 특정 initialization dependency를 줄이는 데 도움이 되었다. 기존 WiFi-guide 방식과 달리 특정 위치를 먼저 강하게 신뢰하지 않고, 여러 anchor 조합에서 생성된 후보를 전체 residual consistency로 평가하였다.

넷째, Huber 기반 IRLS/WLS refinement는 heavy-tailed RTT noise에 대해 비교적 안정적이었다. Huber weight는 큰 residual의 영향을 완화하며, soft residual gating과 함께 outlier anchor가 위치 추정에 미치는 영향을 줄였다.

다섯째, bounded Ridge residual compensation은 robust WLS 이후 남아 있는 systematic residual을 줄이는 데 도움이 되었다. Ridge는 위치를 직접 예측하지 않고, robust WLS 결과의 residual feature를 이용해 제한된 범위 안에서만 보정한다. 이 때문에 direct black-box ML 방식보다 해석 가능성과 안정성이 높다고 판단한다.

하지만 한계도 존재한다. 첫째, subset candidate generation은 계산량이 증가할 수 있다. Candidate 수를 제한하였지만, anchor 조합을 여러 개 평가하는 구조이므로 단순 WLS보다 runtime이 길다. 둘째, train dataset에서 계산한 bias와 sigma는 hidden environment의 measurement 특성이 크게 달라질 경우 효과가 감소할 수 있다. 셋째, Ridge compensation은 선형 보정이므로 복잡한 nonlinear residual pattern을 모두 설명하지는 못한다. 넷째, final-fit에서는 700개 전체를 사용하기 때문에 내부 validation 성능과 final-fit 성능을 구분해서 해석해야 한다.

그럼에도 최종 구조는 물리 기반 robust estimation을 중심으로 하고, Ridge는 bounded 후처리 역할로 제한했기 때문에 direct black-box ML 방식보다 일반화 가능성과 안정성이 높다고 판단한다.

### 4.5 Fairness of Evaluation

이번 실험에서는 hidden test leakage를 방지하기 위해 알고리즘 개발 단계와 final-fit 단계를 구분하였다. 개발 단계에서는 제공된 데이터를 train, validation, test로 나누어 구조와 hyperparameter를 확인하였다. 최종 제출 단계에서는 선택된 구조를 고정한 뒤, 제공된 700개 전체 label 데이터를 사용하여 BIAS, SIGMA, BASE_WEIGHT, Ridge coefficient를 다시 추정하였다.

Baseline 비교에서도 가능한 한 fair한 비교를 유지하고자 하였다. 단순 WLS와 최종 hybrid algorithm을 한 번에 비교하면 추가된 각 구성 요소의 효과를 분리해서 해석하기 어렵다. 따라서 bias calibration, reliability weighting, guide removal, subset consensus, robust IRLS/WLS, bounded Ridge compensation을 단계적으로 검토하였다.

또한 알고리즘 개발 과정에서 사용자 위치를 직접 암기하거나, 제공된 700개 위치 좌표를 lookup하는 방식을 사용하지 않았다. Ridge compensation 역시 d_hat에서 위치를 직접 예측하지 않고, robust WLS가 산출한 p_base의 residual pattern을 기반으로 남은 오차만 보정한다. 이 때문에 baseline과의 비교도 단순 딥러닝 모델과 물리 기반 삼변측량을 무리하게 비교하는 방식이 아니라, 같은 robust localization pipeline 안에서 후처리 보정이 성능에 미치는 영향을 평가하는 방식에 가깝다.

Ridge compensation의 과적합 가능성을 줄이기 위해 다음 설계를 사용하였다.

| Design Choice | Purpose |
|---|---|
| Ridge alpha 적용 | coefficient가 과도하게 커지는 것을 방지 |
| Clip 적용 | 보정값이 비정상적으로 커지는 것을 방지 |
| Lambda mixture 적용 | p_base를 완전히 대체하지 않고 제한적으로 보정 |
| p_base residual feature 사용 | direct position learning보다 물리 기반 구조 유지 |
| model_ridge.npz 저장 | 학습된 상수와 결과 재현성 확보 |
| main.py에서 추가 학습 없음 | hidden test에서 label을 사용하지 않고 추론만 수행 |

### 4.6 Future Work

향후에는 다음과 같은 방향으로 추가 개선이 가능하다고 판단한다.

| Future Work | Expected Effect |
|---|---|
| Adaptive anchor subset selection | 사용자별로 더 안정적인 subset 선택 가능 |
| Distance-bin bias calibration | 가까운 거리와 먼 거리에서 다른 bias를 반영 가능 |
| Sample-wise correction strength | residual uncertainty에 따라 lambda를 다르게 적용 가능 |
| Nonlinear residual correction | Ridge로 설명하기 어려운 nonlinear residual pattern 보정 가능 |
| Runtime optimization | subset candidate generation의 계산량 감소 |

특히 현재 Ridge compensation은 모든 샘플에 동일한 lambda를 적용한다. 향후에는 residual median, candidate spread, robust cost 등을 이용해 샘플별 uncertainty를 계산하고, uncertainty가 큰 샘플에만 보정 강도를 높이는 방식으로 개선할 수 있다. 또한 anchor별 bias를 하나의 상수로 두는 대신 거리 구간별 median bias를 사용하면, 거리 크기에 따라 달라지는 RTT error를 더 잘 반영할 수 있을 것이다.

# Reference

# Reference

[1] P. J. Huber, “Robust Estimation of a Location Parameter,” The Annals of Mathematical Statistics, vol. 35, no. 1, pp. 73–101, 1964.  
Huber loss의 기본 개념을 참고하여, 큰 anchor residual이 WLS 결과를 과도하게 왜곡하지 않도록 residual weighting에 적용하였다.

[2] A. E. Hoerl and R. W. Kennard, “Ridge Regression: Biased Estimation for Nonorthogonal Problems,” Technometrics, vol. 12, no. 1, pp. 55–67, 1970.  
Ridge regression의 정규화 개념을 참고하여, p_base의 남은 오차 p_true - p_base를 안정적으로 학습하는 residual correction에 적용하였다.

[3] F. Wang, H. Tang, and J. Chen, “Survey on NLOS Identification and Error Mitigation for UWB Indoor Positioning,” Electronics, vol. 12, no. 7, 1678, 2023.  
UWB/RTT 기반 실내 측위에서 NLOS로 인한 positive bias와 큰 ranging error가 발생할 수 있다는 배경을 참고하였다.

[4] R. Fan and X. Du, “NLOS Error Mitigation Using Weighted Least Squares and Kalman Filter in UWB Positioning,” arXiv:2205.05939, 2022.  
신뢰도가 낮은 거리 측정값의 영향을 WLS 구조에서 줄일 수 있다는 관련 사례로 참고하였으며, 본 알고리즘은 Kalman filter 대신 anchor subset과 residual consistency를 사용하였다.

