# Indoor Localization Using Anchor Residual-Aware Robust WLS with Bounded Ridge Compensation

# 1. Motivation & Introduction

실내 RTT 기반 측위에서는 거리 측정값이 실제 거리와 상당한 차이를 가지는 경우가 많다. 특히 NLOS 환경에서는 일부 anchor의 거리값이 실제 거리보다 매우 크게 측정되며, 이러한 오차는 일반적인 least squares 기반 위치 추정의 성능을 크게 저하시킨다. 제공된 데이터셋 역시 RTT distance가 전반적으로 positive bias를 가지며, 일부 anchor에서는 매우 큰 outlier가 발생하는 heavy-tailed noise 특성을 보였다.

중간 발표까지 사용하였던 기존 알고리즘은 WiFi-guide 기반 구조였다. 먼저 WiFi distance를 이용해 초기 위치를 추정한 뒤, 해당 위치를 기준으로 UWB gating을 수행하고, 이후 IRLS 기반 weighted least squares를 통해 최종 위치를 refinement하는 방식이었다. 이 구조는 WiFi guide가 안정적일 때는 유효하지만, 초기 guide 위치가 부정확하면 이후 gating 과정도 잘못된 기준점을 따라가게 된다는 한계가 있었다. 즉, guide 위치가 흔들리면 정상 anchor가 제거되거나 반대로 이상치 anchor가 유지될 수 있다.

이 문제를 해결하기 위해 최종 알고리즘에서는 특정 guide 위치를 먼저 신뢰하지 않는 방향으로 구조를 변경하였다. 대신 anchor residual consistency 자체를 기반으로 위치 후보를 만들고, 여러 anchor subset에서 생성된 후보 중 전체 anchor 거리값과 가장 일관적인 후보를 선택하였다. 이후 residual gating과 Tukey-Huber 기반 robust IRLS/WLS를 적용하여 outlier anchor의 영향을 줄였다.

하지만 robust WLS까지 적용한 뒤에도 일부 샘플에서는 위치 오차가 특정 방향으로 반복적으로 남는 현상이 있었다. 이는 단순한 random noise라기보다, anchor 배치와 residual pattern에 의해 발생하는 systematic residual error로 판단하였다. 따라서 최종 단계에서는 기존 물리 기반 추정값을 그대로 대체하지 않고, 그 결과가 남긴 residual pattern만 이용하여 dx, dy를 약하게 보정하는 bounded Ridge residual compensation을 추가하였다.

최종 알고리즘은 다음 흐름으로 구성된다.

| Step | Description |

| 1 | Anchor-wise bias calibration 
| 2 | Anchor reliability estimation 
| 3 | TOP-K anchor selection 
| 4 | Anchor subset candidate generation 
| 5 | Residual-based candidate selection
| 6 | Residual gating
| 7 | Tukey + Huber 기반 robust IRLS/WLS refinement 
| 8 | Bounded Ridge residual compensation 

이 구조의 핵심은 Ridge regression으로 위치를 직접 예측하는 것이 아니다. 먼저 물리적으로 해석 가능한 robust localization pipeline으로 기본 위치 p_base를 계산한 뒤, 그 위치에서 계산되는 residual feature를 이용해 남은 systematic error만 제한적으로 보정한다. 따라서 본 알고리즘은 direct machine learning localization이 아니라, physics-guided robust localization에 lightweight residual learning을 결합한 hybrid 구조이다.

# 2. Algorithm Description

## 2.1 Dataset Analysis

제공된 데이터는 사용자 실제 위치 p, RTT 기반 거리 측정값 d_hat, 그리고 anchor 좌표 p_bs 또는 BS_positions로 구성되어 있다. 한 명의 사용자에 대해 여러 anchor의 거리 측정값이 주어지며, 최종 목표는 각 사용자에 대해 2차원 위치 p_hat을 추정하는 것이다.

거리 측정값을 실제 거리와 비교한 결과 다음과 같은 특징이 관찰되었다.

| Property             | Observation |

| Positive bias        | 측정 거리가 실제 거리보다 크게 나타남 |
| Heavy-tailed noise    | 일부 anchor distance가 매우 크게 튐 |
| Reliability difference | anchor마다 오차 scale과 안정성이 다름 |
| No missing value | NaN 및 Inf가 없어 결측치 대체보다는 오차 보정이 중요함 |

따라서 모든 anchor를 동일한 신뢰도로 사용하는 단순 WLS는 적합하지 않다고 판단하였다. 또한 hidden test set이 별도로 존재하므로, 제공된 데이터의 특정 위치를 암기하는 방식보다 anchor geometry와 residual consistency를 이용하는 일반화 가능한 구조가 필요하였다.

데이터 과적합을 줄이기 위해 제공 데이터는 train, validation, test을 500/100/100으로 나누어 사용하였다. Train dataset은 bias, sigma, reliability, Ridge residual compensation 학습에 사용하였고, validation dataset은 hyperparameter 선택에 사용하였다. Test dataset은 최종 확인용으로만 사용하여 hidden test leakage를 방지하고자 하였다.

## 2.2 Anchor-wise Bias Calibration

RTT 기반 거리 측정값은 실내 환경에서 실제 거리보다 크게 측정되는 경향이 있었다. 이를 완화하기 위해 train dataset에서 anchor별 bias를 계산하였다.

사용자 u와 anchor i에 대해 실제 거리는 다음과 같다.

d_true,i(u) = sqrt((x_u - x_i)^2 + (y_u - y_i)^2)

측정 오차는 다음과 같이 정의하였다.

e_i(u) = d_hat,i(u) - d_true,i(u)

Anchor i의 bias는 평균 대신 median으로 계산하였다.

b_i = median(e_i(u))

Median을 사용한 이유는 일부 anchor에서 매우 큰 outlier가 발생하기 때문이다. Mean은 extreme outlier에 쉽게 흔들리지만, median은 long-tail noise에 상대적으로 강하다.

최종 보정 거리값은 다음과 같이 계산하였다.

d_cal,i = d_i - b_i

또한 보정 후 거리가 비정상적으로 작아지는 경우를 막기 위해 최소 거리 하한을 두었다.

## 2.3 Anchor Reliability Estimation

Anchor마다 measurement stability가 다르므로, train dataset에서 anchor별 residual scale을 계산하였다. Residual scale이 큰 anchor는 신뢰도가 낮다고 판단하고 weight를 낮추었다. 반대로 residual scale이 작은 anchor는 상대적으로 높은 weight를 부여하였다.

Anchor reliability weight는 다음 개념으로 설정하였다.

w_i = 1 / (sigma_i^2 + epsilon)

여기서 sigma_i는 anchor i의 residual scale이며, epsilon은 numerical instability를 방지하기 위한 작은 상수이다. 실제 구현에서는 weight가 지나치게 한쪽으로 치우치지 않도록 정규화된 base weight를 사용하였다.

또한 특정 사용자 샘플에서 보정 거리 d_cal,i가 전체 거리 분포에 비해 지나치게 큰 경우, 해당 anchor는 sample-level outlier일 가능성이 있으므로 reliability를 추가로 감소시켰다.

## 2.4 TOP-K Anchor Selection

모든 anchor를 초기 위치 생성에 그대로 사용하는 대신, reliability가 높은 anchor를 우선적으로 선택하였다. Residual variance가 큰 anchor는 실제 위치와의 consistency가 낮을 가능성이 높기 때문에, 초기 후보 생성 단계에서는 reliability 기준 상위 anchor를 중심으로 사용하였다.

S_top = TopK(w_i)

이 단계는 unstable anchor가 초기 위치 후보를 왜곡하는 것을 줄이는 역할을 한다. 다만 최종 residual cost와 refinement에서는 전체 anchor consistency를 다시 평가하므로, 단순히 일부 anchor만 영구적으로 사용하는 방식은 아니다.

## 2.5 Anchor Subset Candidate Generation

선택된 TOP-K anchor들에 대해 여러 subset 조합을 생성하였다. 각 subset은 독립적인 위치 후보를 만들며, bounded weighted least squares를 이용하여 해당 subset에 가장 잘 맞는 위치를 계산하였다.

Subset S_k에 대한 위치 후보는 다음 목적함수를 최소화하는 위치로 정의된다.

p_k = argmin_p Σ_{i in S_k} w_i * (||p - a_i|| - d_cal,i)^2

여기서 a_i는 anchor i의 좌표이고, d_cal,i는 bias 보정된 거리이다. 각 subset은 서로 다른 anchor 조합을 사용하므로, 특정 anchor configuration에 과도하게 의존하지 않고 여러 가능한 위치 후보를 만들 수 있다.

Candidate 수가 지나치게 많아지면 runtime이 증가하므로, 후보 조합이 많을 경우 reliability 합이 높은 subset을 우선적으로 사용하였다. 이는 hidden test의 실행 시간 제한을 고려한 설계이다.

## 2.6 Residual-based Candidate Selection

생성된 여러 위치 후보 중에서 전체 anchor distance와 가장 일관성이 높은 후보를 초기 위치로 선택하였다.

Candidate p_k에 대한 residual은 다음과 같이 계산된다.

r_i(p_k) = ||p_k - a_i|| - d_cal,i

Anchor마다 residual scale이 다르므로 residual은 sigma_i로 정규화하였다.

r_tilde,i(p_k) = r_i(p_k) / (sigma_i + epsilon)

Candidate cost는 Huber loss 기반으로 계산하였다.

J(p_k) = Σ_i w_i * rho(r_tilde,i(p_k))

여기서 rho는 Huber loss이다. Huber loss를 사용한 이유는 작은 residual에는 제곱 오차처럼 반응하면서도, 큰 residual에는 영향이 선형적으로 제한되기 때문이다.

최종 초기 위치는 다음과 같이 선택된다.

p_0 = argmin_{p_k} J(p_k)

이 방식은 WiFi guide처럼 외부 초기 위치를 직접 신뢰하지 않고, anchor residual consistency 자체로 초기 위치를 선택한다는 점에서 기존 방식과 다르다.

## 2.7 Residual Gating

선택된 초기 위치 p_0를 기준으로 residual gating을 수행하였다. Anchor residual이 지나치게 큰 경우 해당 anchor는 이상치일 가능성이 높다고 판단하였다.

Gate condition은 다음과 같다.

|r_i| <= tau_i

tau_i = GATE_SCALE * sigma_i

Gate를 통과하지 못한 anchor는 완전히 제거하지 않고 weight를 매우 작게 낮추었다. 완전 제거 대신 soft down-weighting을 사용한 이유는 hidden test에서 특정 anchor가 우연히 큰 residual을 보이더라도, 해당 anchor의 정보를 완전히 잃는 것을 방지하기 위해서이다.

만약 gate를 통과한 anchor 수가 너무 적으면 WLS가 불안정해질 수 있으므로, residual이 작은 anchor를 최소 개수 이상 유지하도록 하였다.

## 2.8 Tukey + Huber Robust IRLS/WLS Refinement

Residual gating 이후에는 IRLS 기반 robust weighted least squares를 반복 수행하였다. 이 단계에서는 residual이 큰 anchor의 영향을 반복적으로 줄이며 위치를 refine한다.

Tukey weight는 다음 개념으로 계산된다.

u_i = r_i / (c * sigma_i)

w_tukey,i = (1 - u_i^2)^2, if |u_i| < 1

w_tukey,i = 0, if |u_i| >= 1

Tukey weight는 residual이 매우 큰 anchor의 영향을 거의 제거하는 역할을 한다.

Huber weight는 다음과 같이 계산된다.

w_huber,i = 1, if |r_i| <= delta

w_huber,i = delta / |r_i|, if |r_i| > delta

Huber weight는 큰 residual의 영향을 줄이되 완전히 0으로 만들지는 않는 역할을 한다.

최종 weight는 다음과 같이 계산하였다.

w_final,i = w_base,i * w_tukey,i * w_huber,i

이후 bounded WLS를 반복 수행하여 위치를 갱신하였다.

p_base = argmin_p Σ_i w_final,i * (||p - a_i|| - d_cal,i)^2

반복은 위치 변화량이 충분히 작아지거나 최대 반복 횟수에 도달하면 종료하였다. 이 결과 p_base는 물리 기반 robust localization pipeline이 산출한 기본 위치이다.

## 2.9 Bounded Ridge Residual Compensation

Robust WLS 결과만으로도 outlier 영향을 줄일 수 있었지만, validation 결과를 분석하면 일부 샘플에서 residual pattern에 따른 systematic error가 남아 있었다. 따라서 최종 단계에서는 Ridge regression을 이용하여 p_base의 남은 위치 오차 dx, dy를 보정하였다.

중요한 점은 Ridge regression이 d_hat에서 위치를 직접 예측하지 않는다는 것이다. Ridge는 p_base와 p_base에서 계산한 residual feature를 입력으로 받아, p_base가 실제 위치에서 얼마나 벗어났는지만 학습한다.

Ridge feature에는 다음 정보가 포함된다.

| Feature Group | Meaning |

| p_base_x, p_base_y | 기존 robust WLS가 산출한 위치 |
| residual mean, median, std | p_base 기준 anchor residual의 전체 경향 |
| residual min, max | 특정 방향의 큰 불일치 여부 |
| absolute residual statistics | 전체 residual 크기와 outlier 정도 |
| top residual statistics | 가장 큰 residual들의 영향 |
| robust cost | p_base의 anchor consistency 정도 |
| d_cal statistics | 보정 거리값의 분포 |
| positive/negative residual count | residual 방향성의 불균형 |

Target은 다음과 같이 정의하였다.

delta_true = p_true - p_base

Ridge regression은 다음 관계를 학습한다.

delta_ridge = X_norm * W + b

여기서 X_norm은 feature를 train dataset의 mean과 standard deviation으로 정규화한 값이다. Ridge는 다음 objective를 최소화한다.

min_W ||Y - XW||^2 + alpha * ||W||^2

최종 위치는 Ridge 보정값을 그대로 적용하지 않고, clip과 lambda mixture를 적용하였다.

delta_clipped = clip(delta_ridge, -C, C)

p_final = p_base + lambda * delta_clipped

이 bounded correction 구조를 사용한 이유는 hidden test에서 Ridge 보정이 과하게 튀는 것을 방지하기 위해서이다. 즉, 최종 위치는 물리 기반 p_base를 중심으로 제한된 범위 안에서만 수정된다.

## 2.10 Hyperparameter Selection

Hyperparameter selection은 hidden test leakage를 방지하기 위해 train dataset과 validation dataset만을 사용하여 수행하였다. 기존 robust WLS 관련 parameter는 validation RMSE 기준으로 선택하였다. Ridge residual compensation의 alpha, lambda, clip 역시 train에서 학습하고 validation에서 선택하였다.

| Parameter | Role |

| TOP_K | 초기 후보 생성에 사용할 신뢰도 상위 anchor 개수 |
| GATE_SCALE | residual gating threshold scale |
| HUBER_DELTA | Huber loss 및 Huber weight threshold |
| TUKEY_SCALE | Tukey weight normalization scale |
| Ridge alpha | Ridge coefficient regularization strength |
| Ridge lambda | Ridge 보정값을 최종 위치에 반영하는 비율 |
| Ridge clip | Ridge 보정량의 최대 허용 범위 |

최종 선택된 parameter는 다음과 같다.

| Parameter | Selected Value |

| TOP_K | 8 |
| GATE_SCALE | 3.5 |
| HUBER_DELTA | 12.0 |
| TUKEY_SCALE | 6.0 |
| Ridge alpha | 10.0 |
| Ridge lambda | 0.7 |
| Ridge clip | 10.0 |

# 3. Agent AI Usage

이번 실험에서는 ChatGPT를 포함한 Agent AI를 알고리즘 설계 보조 도구로 활용하였다.

AI는 주로 robust estimation 관련 이론 정리, residual weighting 방식 비교, IRLS 구조 해석, hyperparameter tuning 방향 탐색, 보고서 구조 정리 등에 사용하였다. 또한 Huber loss, Tukey weight, residual gating, anchor reliability weighting과 같은 개념을 현재 데이터셋에 어떻게 적용할 수 있을지 비교·정리하는 과정에서 활용하였다.

초기 단계에서는 기존 WiFi-guide 기반 UWB gating 구조의 장점과 한계를 정리하는 데 AI를 참고하였다. 이후 교수님 피드백과 데이터 분석 결과를 바탕으로, WiFi guide에 의존하지 않고 anchor residual consistency를 이용하는 방향을 검토하였다. 이 과정에서 AI는 grid search, anchor subset consensus, RANSAC-like candidate selection, robust WLS refinement, residual correction 등 여러 후보 방법의 장단점을 비교하는 보조 역할을 하였다.

하지만 실제 알고리즘 구조 선택과 실험 방향 결정은 직접 수행하였다. RTT distance의 positive bias 특성 분석, anchor별 residual scale 분석, TOP-K anchor selection 구조 채택, subset candidate generation 설계, residual gating 적용 여부, Tukey-Huber IRLS/WLS 구성, validation 기반 parameter tuning 및 RMSE/MAE 비교 실험은 반복적인 실험 결과를 기준으로 직접 진행하였다.

또한 실제 validation 결과와 hidden test generalization 가능성을 기준으로 구조를 수정하거나 제외하였다. 예를 들어 계산량이 큰 grid search 구조는 검토 단계에서 제외하고, 최종적으로는 anchor subset 기반 initialization과 robust residual refinement 구조를 선택하였다.

# 4. Result & Discussion

## 4.1 Evaluation Method

위치 추정 성능은 RMSE와 MAE를 기준으로 평가하였다.

RMSE = sqrt((1/N) * Σ_i ||p_i - p_hat,i||^2)

MAE = (1/N) * Σ_i ||p_i - p_hat,i||

Train dataset은 bias, sigma, base weight, Ridge residual compensation 학습에 사용하였다. Validation dataset은 TOP_K, GATE_SCALE, HUBER_DELTA, TUKEY_SCALE, Ridge alpha, Ridge lambda, Ridge clip을 선택하는 데 사용하였다. Test dataset은 최종 확인용으로만 사용하였다.

본 알고리즘은 hidden test set에 대해 평가되므로, 제공된 데이터에만 과도하게 맞는 구조는 피하고자 하였다. 특히 Ridge compensation은 train에서만 coefficient를 학습하고, validation에서는 보정 강도와 regularization을 선택하였다. 또한 보정량에 clip과 lambda를 적용하여 hidden environment에서 과도한 correction이 발생하지 않도록 하였다.

## 4.2 Quantitative Result

최종 알고리즘의 train/validation 성능은 다음과 같다.

| Robust WLS base | 11.3930 | N/A | 11.7439 | N/A |
| Robust WLS + bounded Ridge compensation | 10.5566 | 8.2433 | 10.9048 | 8.1319 |

Ridge compensation을 추가했을 때 RMSE 개선량은 다음과 같다.

| Dataset | Base RMSE | Final RMSE | Improvement |

| Train | 11.3930 | 10.5566 | 0.8364 |
| Validation | 11.7439 | 10.9048 | 0.8391 |

Train과 validation에서 RMSE 개선량이 거의 비슷하게 나타났기 때문에, Ridge compensation이 train data만 과도하게 외웠다고 보기는 어렵다. 다만 lambda와 clip이 비교적 강한 값으로 선택되었기 때문에, hidden test에서는 bounded correction이 과하게 작동하지 않도록 clip과 lambda mixture를 유지하였다.

## 4.3 Baseline Comparison and Development Process

이번 실험은 단순 WLS에서 시작하여, 단계적으로 robust estimation 구조를 추가하면서 발전하였다. 초기에는 모든 anchor를 동일한 신뢰도로 사용하는 basic WLS를 고려하였다. 하지만 RTT distance에는 positive bias와 heavy-tailed noise가 존재하므로, 일부 이상치 anchor가 위치 추정 결과를 크게 왜곡하였다.

다음 단계에서는 anchor-wise bias calibration을 적용하였다. Train dataset에서 각 anchor의 median error를 계산하고 이를 측정 거리에서 제거함으로써, RTT distance가 실제 거리보다 지속적으로 크게 측정되는 현상을 완화하고자 하였다.

이후 anchor reliability weighting을 추가하였다. Anchor마다 residual scale이 다르게 나타났기 때문에, variance가 큰 anchor의 weight를 낮추고 안정적인 anchor의 weight를 높였다. 이를 통해 unstable anchor가 위치 추정에 미치는 영향을 줄일 수 있었다.

처음에는 coarse/fine grid search 기반 global search도 고려하였다. 전체 공간에서 residual consistency가 가장 좋은 위치를 탐색하는 방식은 guide dependency를 줄인다는 장점이 있었지만, 계산량이 증가하고 validation 성능 개선이 제한적이었다. 이에 따라 최종적으로는 reliability가 높은 anchor subset을 여러 개 만들고, 각 subset에서 생성된 후보 중 전체 anchor residual cost가 가장 작은 후보를 선택하는 anchor subset consensus 방식을 사용하였다.

마지막으로 robust WLS 결과에서 남는 systematic residual을 줄이기 위해 bounded Ridge residual compensation을 추가하였다. 이 단계는 위치를 직접 예측하는 ML 모델이 아니라, 기존 robust WLS 결과 p_base의 residual feature를 이용해 남은 dx, dy만 제한적으로 보정하는 구조이다.

전체 개발 과정은 다음과 같이 정리할 수 있다.

| Stage | Main Idea | Reason for Adoption or Rejection |

| Basic WLS | 모든 anchor 동일 사용 | outlier에 취약하여 한계 존재 |
| Bias calibration | anchor별 median bias 보정 | RTT positive bias 완화에 필요 |
| Reliability weighting | anchor별 residual scale 기반 weight | unstable anchor 영향 감소 |
| WiFi-guide fusion | guide 위치 기준 gating | guide가 부정확하면 gating도 흔들림 |
| Grid-based search | 전체 공간 residual consistency 탐색 | 계산량 증가 및 성능 개선 제한 |
| Anchor subset consensus | 여러 anchor subset 후보 생성 후 residual cost로 선택 | guide 의존성 감소 및 robust initialization 가능 |
| Tukey-Huber IRLS/WLS | residual 기반 반복 reweighting | heavy-tailed noise에 대응 |
| Bounded Ridge compensation | p_base의 residual pattern으로 dx, dy 제한 보정 | robust WLS 이후 남는 systematic error 감소 |

이 과정은 단순히 RMSE 숫자만 줄이는 방향이 아니라, hidden environment에서도 일반화 가능한 구조를 찾는 과정이었다. 특히 Ridge compensation을 마지막에 추가하였지만, 전체 알고리즘의 중심은 여전히 anchor geometry와 residual consistency를 이용한 물리 기반 robust estimation이다.

## 4.4 Discussion About the Proposed Method
이 알고리즘은 특정 guide 위치 dependency를 줄이고, anchor residual consistency 자체를 기반으로 위치를 추정하도록 설계하였다. 이 방향은 기존 WiFi-guide 방식의 한계를 보완하는 데 적절했다고 판단한다.

첫째, anchor-wise bias calibration은 RTT distance의 지속적인 positive bias를 완화하는 데 도움이 되었다. 실제 거리보다 크게 측정되는 경향을 anchor별 median bias로 제거함으로써, 이후 WLS가 더 합리적인 거리값을 사용하도록 만들었다.

둘째, anchor reliability weighting은 unstable anchor의 영향을 줄이는 데 효과적이었다. 모든 anchor를 동일하게 사용하는 것보다 residual scale 기반 weighting이 더 안정적인 결과를 제공하였다.

셋째, anchor subset consensus 구조는 특정 initialization dependency를 줄이는 데 도움이 되었다. 기존 WiFi-guide 방식과 달리 특정 위치를 먼저 강하게 신뢰하지 않고, 여러 anchor 조합에서 생성된 후보를 전체 residual consistency로 평가하였다.

넷째, Tukey + Huber 기반 IRLS/WLS refinement는 heavy-tailed RTT noise에 대해 비교적 안정적이었다. Tukey weight는 매우 큰 residual을 강하게 줄이고, Huber weight는 중간 크기의 outlier 영향을 완화하였다.

다섯째, bounded Ridge residual compensation은 robust WLS 이후 남아 있는 systematic residual을 줄이는 데 도움이 되었다. Train과 validation에서 RMSE 개선량이 비슷하게 나타났기 때문에, 단순히 train data만 외운 correction이라기보다는 residual pattern에 대한 일반적인 보정 효과가 있었다고 판단하였다.

하지만 한계도 존재한다. 첫째, subset candidate generation은 계산량이 증가할 수 있다. Candidate 수를 제한하였지만, anchor 조합을 여러 개 평가하는 구조이므로 단순 WLS보다 runtime이 길다. 둘째, train dataset에서 계산한 bias와 sigma는 hidden environment의 measurement 특성이 크게 달라질 경우 효과가 감소할 수 있다. 셋째, Ridge compensation은 선형 보정이므로 복잡한 nonlinear residual pattern을 모두 설명하지는 못한다. 넷째, lambda와 clip이 validation 기준으로 선택되었기 때문에 validation overfitting 가능성을 완전히 배제할 수는 없다.

그럼에도 최종 구조는 물리 기반 robust estimation을 중심으로 하고, Ridge는 bounded 후처리 역할로 제한했기 때문에 direct black-box ML 방식보다 해석 가능성과 안정성이 높다고 판단한다.

## 4.5 Fairness of Evaluation

이번 실험에서는 hidden test leakage를 방지하기 위해 train, validation, test 역할을 구분하였다. Train dataset은 bias, sigma, base weight, Ridge coefficient 학습에 사용하였고, validation dataset은 hyperparameter 선택에 사용하였다. Test dataset은 최종 확인용으로만 사용하였다.

Baseline 비교에서도 가능한 한 fair한 비교를 유지하고자 하였다. 단순 WLS와 최종 hybrid algorithm을 한 번에 비교하면 추가된 각 구성 요소의 효과를 분리해서 해석하기 어렵다. 따라서 bias calibration, reliability weighting, guide removal, subset consensus, robust IRLS/WLS, bounded Ridge compensation을 단계적으로 검토하였다.

또한 알고리즘 개발과정에서 사용자 위치를 직접 암기하거나, 제공된 700개 위치 좌표를 lookup하는 방식을 사용하지 않았다. Ridge compensation 역시 d_hat에서 위치를 직접 예측하지 않고, robust WLS가 산출한 p_base의 residual pattern을 기반으로 남은 오차만 보정한다. 이 때문에 baseline과의 비교도 단순 딥러닝 모델과 물리 기반 삼변측량을 무리하게 비교하는 방식이 아니라, 같은 robust localization pipeline 안에서 후처리 보정이 성능에 미치는 영향을 평가하는 방식에 가깝다.

Ridge compensation의 과적합 가능성을 줄이기 위해 다음 설계를 사용하였다.

| Design Choice | Purpose |

| Train에서만 Ridge coefficient 학습 | Validation 정답을 직접 학습하지 않음 |
| Validation으로 alpha, lambda, clip 선택 | Hidden test를 직접 사용하지 않음 |
| Ridge alpha 적용 | coefficient가 과도하게 커지는 것을 방지 |
| Clip 적용 | 보정값이 비정상적으로 커지는 것을 방지 |
| Lambda mixture 적용 | p_base를 완전히 대체하지 않고 제한적으로 보정 |
| p_base residual feature 사용 | direct position learning보다 물리 기반 구조 유지 |

## 4.6 Future Work

향후에는 다음과 같은 방향으로 추가 개선이 가능하다고 판단한다.

| Future Work | Expected Effect |

| Adaptive anchor subset selection | 사용자별로 더 안정적인 subset을 선택 가능 |
| Distance-bin bias calibration | 가까운 거리와 먼 거리에서 다른 bias를 반영 가능 |
| Sample-wise correction strength | residual uncertainty에 따라 lambda를 다르게 적용 가능 |
| Nonlinear residual correction | Ridge로 설명하기 어려운 nonlinear residual pattern 보정 가능 |
| Runtime optimization | subset candidate generation의 계산량 감소 |

특히 현재 Ridge compensation은 모든 샘플에 동일한 lambda를 적용한다. 향후에는 residual median, candidate spread, robust cost 등을 이용해 샘플별 uncertainty를 계산하고, uncertainty가 큰 샘플에만 보정 강도를 높이는 방식으로 개선할 수 있다. 또한 anchor별 bias를 하나의 상수로 두는 대신 거리 구간별 median bias를 사용하면, 거리 크기에 따라 달라지는 RTT error를 더 잘 반영할 수 있을 것이다.
