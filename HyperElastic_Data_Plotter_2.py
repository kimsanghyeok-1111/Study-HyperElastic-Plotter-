# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:20:53 2026

@author: DESKTOP-STUDENT
"""

"""
Hyperelastic Model Fitting (W 기준 정의 -> sigma_nom = dW/d(lambda) 자동미분 방식)
--------------------------------------------------------------------------
핵심 아이디어
  - 모든 모델을 "Strain Energy W" 하나만 정의한다 (다른 건 아무것도 손대지 않음).
  - I1(lambda), I2(lambda) 관계식을 미리 대입해 W를 lambda만의 함수로 만든 뒤,
        sigma_nom = dW/d(lambda)
    단 한 줄로 미분해서 Nominal Stress 식을 자동으로 얻는다.
  - Ogden 모델도 예외 없이 이 틀에 그대로 들어간다
    (I1,I2 invariant 대신 주스트레치(lambda1,lambda2,lambda3)로 W를 적고
     비압축 조건 lambda2=lambda3=lambda^-1/2 을 대입한 뒤 동일하게 d/d(lambda) 미분).
  - 즉, 5개 모델 전부 "W = ... " 한 줄 + 미분 한 줄로 통일됨.

  [검증] sigma_nom = dW/d(lambda) 는 연쇄법칙으로 풀면
         dW/dI1 * dI1/d(lambda) + dW/dI2 * dI2/d(lambda)
       = 2*(lambda - lambda^-2) * [ dW/dI1 + (1/lambda)*dW/dI2 ]
  가 되어, 기존에 쓰던 표준 공식과 정확히 동일한 결과를 준다 (더 짧게 쓰는 것 뿐).

입력 데이터 : HyperElastic_Test_data.ods
    - Column A : Nominal Stress
    - Column B : Nominal Strain
    - Row 1    : 헤더, Row 2~ : 데이터

필요 패키지 : pandas, numpy, scipy, sympy, matplotlib, odfpy
    pip install pandas numpy scipy sympy matplotlib odfpy
"""

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# =========================================================
# 0. 사용자 설정
# =========================================================
FILE_PATH = "HyperElastic_Test_data.ods"
SHEET_NAME = "시트1"
STRESS_COL = 0
STRAIN_COL = 1
DATA_ROW_START = 2   # 0-based, 헤더 다음 행부터


# =========================================================
# 1. Test Data 로드
# =========================================================
def load_test_data(path, sheet, srow, stress_col, strain_col):
    df = pd.read_excel(path, engine="odf", sheet_name=sheet, header=None)
    data = df.iloc[srow:, [stress_col, strain_col]].dropna().astype(float)
    stress = data.iloc[:, 0].values
    strain = data.iloc[:, 1].values
    return strain, stress


# =========================================================
# 2. 심볼 정의 : lambda 하나만 독립변수로 사용
#    I1(lambda), I2(lambda) : 단축인장 + 비압축성 조건 결과
# =========================================================
lam = sp.symbols("lambda", positive=True)

I1_of_lam = lam**2 + 2 / lam          # I_1 = lambda^2 + 2/lambda
I2_of_lam = 2 * lam + 1 / lam**2      # I_2 = 2*lambda + 1/lambda^2

C10, C01, C20, C11, C02, C30 = sp.symbols("C10 C01 C20 C11 C02 C30")
mu0, lam_m = sp.symbols("mu0 lambda_m", positive=True)
mu1, a1 = sp.symbols("mu1 alpha1")

# Arruda-Boyce 8-chain 급수계수 (고정 상수, i=1..5)
_AB_CI = [sp.Rational(1, 2), sp.Rational(1, 20), sp.Rational(11, 1050),
          sp.Rational(19, 7000), sp.Rational(519, 673750)]


# =========================================================
# 3. sigma_nom = dW/d(lambda)  -- 모든 모델 공통, 단 한 줄
# =========================================================
def sigma_nom_from_W(W_lambda):
    """W_lambda : 이미 lambda 만의 함수로 정리된 Strain Energy 식"""
    return sp.diff(W_lambda, lam)


# =========================================================
# 4. 모델별 Strain Energy  W = ...  만 정의
#    (I1, I2 자리에 위의 I1_of_lam, I2_of_lam 을 바로 대입해서 작성)
# =========================================================

# ① Mooney-Rivlin :  W = C10*(I1-3) + C01*(I2-3)
W_mooney_rivlin = C10 * (I1_of_lam - 3) + C01 * (I2_of_lam - 3)

# ② Yeoh :  W = C10*(I1-3) + C20*(I1-3)^2 + C30*(I1-3)^3
W_yeoh = (
    C10 * (I1_of_lam - 3)
    + C20 * (I1_of_lam - 3) ** 2
    + C30 * (I1_of_lam - 3) ** 3
)

# ③ Arruda-Boyce :  W = mu0 * sum_i [Ci / lam_m^(2i-2)] * (I1^i - 3^i)
W_arruda_boyce = mu0 * sum(
    _AB_CI[i - 1] / lam_m ** (2 * i - 2) * (I1_of_lam ** i - 3 ** i)
    for i in range(1, 6)
)

# ④ Ogden (N=1) :  W = (mu1/a1) * (lambda1^a1 + lambda2^a1 + lambda3^a1 - 3)
#    비압축 단축 조건 lambda1=lambda, lambda2=lambda3=lambda^-1/2 대입
W_ogden_n1 = (mu1 / a1) * (lam ** a1 + 2 * lam ** (-a1 / 2) - 3)

# ⑤ Polynomial (N=2, 5-parameter)
x = I1_of_lam - 3
y = I2_of_lam - 3
W_polynomial_n2 = (
    C10 * x + C01 * y
    + C20 * x ** 2 + C11 * x * y + C02 * y ** 2
)

MODELS = {
    "Mooney-Rivlin": dict(
        W=W_mooney_rivlin, params=[C10, C01],
        p0=[0.1, 0.1], bounds=None,
    ),
    "Yeoh": dict(
        W=W_yeoh, params=[C10, C20, C30],
        p0=[0.1, 0.01, 0.001], bounds=None,
    ),
    "Arruda-Boyce": dict(
        W=W_arruda_boyce, params=[mu0, lam_m],
        p0=[0.3, 5.0], bounds=([1e-3, 1.5], [1e4, 50]),
    ),
    "Ogden(N=1)": dict(
        W=W_ogden_n1, params=[mu1, a1],
        p0=[300, 2], bounds=None,
    ),
    "Polynomial(N=2)": dict(
        W=W_polynomial_n2, params=[C10, C01, C20, C11, C02],
        p0=[0.1, 0.1, 0.01, 0.01, 0.01], bounds=None,
    ),
}


# =========================================================
# 5. sympy 기호식 -> numpy 계산 함수 (lambdify)
# =========================================================
def make_numeric_func(W_lambda, params):
    sigma_expr = sigma_nom_from_W(W_lambda)     # sigma_nom = dW/d(lambda)
    f_lambd = sp.lambdify((lam, *params), sigma_expr, "numpy")

    def func(lam_arr, *p):
        return f_lambd(lam_arr, *p)

    return func, sigma_expr


# =========================================================
# 6. Fitting 실행
# =========================================================
def fit_all_models(lam_data, stress_data, models):
    results = {}
    for name, spec in models.items():
        func, sigma_expr = make_numeric_func(spec["W"], spec["params"])
        kwargs = dict(p0=spec["p0"], maxfev=50000)
        if spec["bounds"] is not None:
            kwargs["bounds"] = spec["bounds"]
        popt, _ = curve_fit(func, lam_data, stress_data, **kwargs)
        pred = func(lam_data, *popt)
        ss_res = np.sum((stress_data - pred) ** 2)
        ss_tot = np.sum((stress_data - np.mean(stress_data)) ** 2)
        r2 = 1 - ss_res / ss_tot
        results[name] = dict(
            func=func, popt=popt, r2=r2,
            param_names=[str(p) for p in spec["params"]],
            sigma_expr=sigma_expr,
        )
    return results


def main():
    strain, stress = load_test_data(
        FILE_PATH, SHEET_NAME, DATA_ROW_START, STRESS_COL, STRAIN_COL
    )
    lam_data = 1.0 + strain

    results = fit_all_models(lam_data, stress, MODELS)

    print(f"{'Model':<18}{'R2':>10}   Parameters")
    print("-" * 80)
    for name, r in results.items():
        param_str = ", ".join(
            f"{n}={v:.4f}" for n, v in zip(r["param_names"], r["popt"])
        )
        print(f"{name:<18}{r['r2']:>10.5f}   {param_str}")

    # 참고: sympy가 자동으로 유도한 sigma_nom(=dW/dλ) 수식 확인
    print("\n--- sympy가 W로부터 자동 유도한 sigma_nom 식 (참고) ---")
    for name, r in results.items():
        print(f"[{name}]  sigma_nom = {sp.simplify(r['sigma_expr'])}")

    # ---- Plot ----
    lam_fine = np.linspace(1.0, lam_data.max() * 1.02, 300)
    plt.figure(figsize=(8, 6))
    plt.plot(strain, stress, "ko", ms=6, label="Test Data (Uniaxial)", zorder=5)

    for name, r in results.items():
        pred_fine = r["func"](lam_fine, *r["popt"])
        plt.plot(lam_fine - 1, pred_fine, "-", label=f"{name} (R\u00b2={r['r2']:.4f})")

    plt.xlabel("Nominal Strain")
    plt.ylabel("Nominal Stress")
    plt.title("Hyperelastic Model Fitting (sigma_nom = dW/d(lambda))")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("hyperelastic_fitting_W.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()