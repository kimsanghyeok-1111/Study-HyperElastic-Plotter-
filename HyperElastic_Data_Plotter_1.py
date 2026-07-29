# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:11:55 2026

@author: DESKTOP-STUDENT
"""

"""
Hyperelastic Model Fitting (sigma_nom 수식을 sympy로 기호화 -> 자동미분 방식)
--------------------------------------------------------------------------
핵심 아이디어
  - 각 모델을 W(I1, I2) "수식" 그대로 정의만 하면,
    sigma_nom = 2*(lam - lam**-2) * [dW/dI1 + (1/lam)*dW/dI2]
    는 sympy로 자동 미분해서 만들어준다.
  - 즉, 모델별로 dW/dI1, dW/dI2를 손으로 유도해서 하드코딩할 필요가 없음.
  - 새 모델을 추가하고 싶으면 MODELS 딕셔너리에 W 수식만 한 줄 추가하면 됨.

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
# 2. sigma_nom 공식을 "기호식"으로 정의
#    - lam, I1, I2 는 sympy 심볼
#    - I1(lam), I2(lam) : 단축인장(비압축) 관계식
#    - sigma_nom(W) : W(I1,I2)를 받아 미분 후 lambda 대입한 결과를 리턴하는 함수
# =========================================================
lam = sp.symbols("lambda", positive=True)
I1s, I2s = sp.symbols("I1 I2")

I1_expr = lam**2 + 2 / lam          # I1(lambda)
I2_expr = 2 * lam + 1 / lam**2      # I2(lambda)


def sigma_nom_expr(W):
    """
    W : I1s, I2s 로 표현된 Strain Energy 기호식
    return : lambda 만의 함수로 정리된 sigma_nom 기호식
    """
    dWdI1 = sp.diff(W, I1s)
    dWdI2 = sp.diff(W, I2s)
    sigma = 2 * (lam - lam**-2) * (dWdI1 + dWdI2 / lam)
    # I1, I2 자리에 lambda로 표현된 식을 대입
    sigma = sigma.subs({I1s: I1_expr, I2s: I2_expr})
    return sp.simplify(sigma)


# =========================================================
# 3. 모델별 Strain Energy W(I1, I2) 정의
#    -> 아래 딕셔너리에 모델 이름 : (W수식, 파라미터리스트, 초기값, bounds) 만 넣으면 끝
# =========================================================
C10, C01, C20, C11, C02, C30 = sp.symbols("C10 C01 C20 C11 C02 C30")
mu0, lam_m = sp.symbols("mu0 lambda_m", positive=True)
mu1, mu2, mu3, a1, a2, a3 = sp.symbols("mu1 mu2 mu3 alpha1 alpha2 alpha3")

# Arruda-Boyce 8-chain 급수계수 (고정 상수)
_AB_CI = [sp.Rational(1, 2), sp.Rational(1, 20), sp.Rational(11, 1050),
          sp.Rational(19, 7000), sp.Rational(519, 673750)]

W_mooney_rivlin = C10 * (I1s - 3) + C01 * (I2s - 3)

W_yeoh = C10 * (I1s - 3) + C20 * (I1s - 3) ** 2 + C30 * (I1s - 3) ** 3

W_arruda_boyce = mu0 * sum(
    _AB_CI[i - 1] / lam_m ** (2 * i - 2) * (I1s ** i - 3 ** i) for i in range(1, 6)
)

W_polynomial_n2 = (
    C10 * (I1s - 3) + C01 * (I2s - 3)
    + C20 * (I1s - 3) ** 2 + C11 * (I1s - 3) * (I2s - 3) + C02 * (I2s - 3) ** 2
)

# Ogden 은 I1,I2가 아닌 주스트레치 기반이라 sigma_nom을 직접 기호식으로 정의
# (I1/I2 경유 미분 방식과 결과는 동일함 - Ogden 원 논문의 stretch-invariant 유도식)
sigma_ogden_n1 = mu1 * (lam ** (a1 - 1) - lam ** (-a1 / 2 - 1))

MODELS = {
    "Mooney-Rivlin": dict(
        sigma_expr=sigma_nom_expr(W_mooney_rivlin),
        params=[C10, C01],
        p0=[0.1, 0.1],
        bounds=None,
    ),
    "Yeoh": dict(
        sigma_expr=sigma_nom_expr(W_yeoh),
        params=[C10, C20, C30],
        p0=[0.1, 0.01, 0.001],
        bounds=None,
    ),
    "Arruda-Boyce": dict(
        sigma_expr=sigma_nom_expr(W_arruda_boyce),
        params=[mu0, lam_m],
        p0=[0.3, 5.0],
        bounds=([1e-3, 1.5], [1e4, 50]),
    ),
    "Ogden(N=1)": dict(
        sigma_expr=sigma_ogden_n1,
        params=[mu1, a1],
        p0=[300, 2],
        bounds=None,
    ),
    "Polynomial(N=2)": dict(
        sigma_expr=sigma_nom_expr(W_polynomial_n2),
        params=[C10, C01, C20, C11, C02],
        p0=[0.1, 0.1, 0.01, 0.01, 0.01],
        bounds=None,
    ),
}


# =========================================================
# 4. sympy 기호식 -> numpy로 계산 가능한 함수로 변환 (lambdify)
#    curve_fit이 요구하는 func(lam_array, *params) 형태로 래핑
# =========================================================
def make_numeric_func(sigma_expr, params):
    f_lambd = sp.lambdify((lam, *params), sigma_expr, "numpy")

    def func(lam_arr, *p):
        return f_lambd(lam_arr, *p)

    return func


# =========================================================
# 5. Fitting 실행
# =========================================================
def fit_all_models(lam_data, stress_data, models):
    results = {}
    for name, spec in models.items():
        func = make_numeric_func(spec["sigma_expr"], spec["params"])
        kwargs = dict(p0=spec["p0"], maxfev=50000)
        if spec["bounds"] is not None:
            kwargs["bounds"] = spec["bounds"]
        popt, _ = curve_fit(func, lam_data, stress_data, **kwargs)
        pred = func(lam_data, *popt)
        ss_res = np.sum((stress_data - pred) ** 2)
        ss_tot = np.sum((stress_data - np.mean(stress_data)) ** 2)
        r2 = 1 - ss_res / ss_tot
        results[name] = dict(func=func, popt=popt, r2=r2,
                              param_names=[str(p) for p in spec["params"]])
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

    # ---- Plot ----
    lam_fine = np.linspace(1.0, lam_data.max() * 1.02, 300)
    plt.figure(figsize=(8, 6))
    plt.plot(strain, stress, "ko", ms=6, label="Test Data (Uniaxial)", zorder=5)

    for name, r in results.items():
        pred_fine = r["func"](lam_fine, *r["popt"])
        plt.plot(lam_fine - 1, pred_fine, "-", label=f"{name} (R\u00b2={r['r2']:.4f})")

    plt.xlabel("Nominal Strain")
    plt.ylabel("Nominal Stress")
    plt.title("Hyperelastic Model Fitting (sigma_nom via sympy auto-diff)")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("hyperelastic_fitting_sympy.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()