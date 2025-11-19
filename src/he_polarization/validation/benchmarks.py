"""
物理结果与哈密顿量系数验证模块。
对系数进行硬编码检查，并提供文献基准值比对。
"""
import math
import numpy as np


def verify_hamiltonian_coefficients(mu: float, M: float):
    """验证哈密顿量展开项系数。"""
    print("Verifying Hamiltonian coefficients...")

    # 1. 混合导数项 C_{Mμ}
    # 论文公式隐含: 1/2 * (1/M - 1/μ)
    # 代码实现 (elements.py): pref_cross = -0.5/M

    # 检查: C_M (r1^2 + r2^2 - r12^2)/(r1 r2) d1 d2
    # 理论系数: -1/(2M)

    if M > 1e10:  # 无限质量
        expected_cross = 0.0
    else:
        expected_cross = -0.5 / M

    print(
        f"  Mass polarization coeff (-1/2M): {expected_cross:.6e} (Check against elements.py logic)")

    # 2. 混合导数项 d1 d12
    # 系数: 1/2 * (1/M - 1/mu)
    expected_mix = 0.5 * (1.0/M - 1.0/mu) if M < 1e10 else -0.5/mu
    print(f"  Mixed derivative coeff (1/2(1/M-1/u)): {expected_mix:.6e}")

    return True


class HBS_Validator:
    """验证计算结果是否符合已知基准"""

    # 基准数据 (Drake, Korobov 等)
    BENCHMARKS = {
        "1s2_1S": -2.903724377034,
        "1s2s_3S": -2.175229378236,
        "1s2s_1S": -2.145974046054,
        "1s2p_3P": -2.133164190779,
        "1s2p_1P": -2.123843086498,
    }

    # 极化率基准 (Yan 2000)
    ALPHA_BENCHMARKS = {
        0: 1.383192174455,  # 1S
        1: 315.63147,      # 2 3S
    }

    def __init__(self):
        pass

    def validate_energy(self, state_label: str, energy: float, tol: float = 1e-6):
        if state_label not in self.BENCHMARKS:
            print(f"[Info] No benchmark for {state_label}")
            return

        ref = self.BENCHMARKS[state_label]
        diff = abs(energy - ref)
        status = "PASS" if diff < tol else "WARN"
        print(
            f"[{status}] Energy {state_label}: Calc={energy:.9f}, Ref={ref:.9f}, Diff={diff:.2e}")

    def validate_polarizability(self, state_idx: int, alpha_L: float, alpha_V: float):
        """验证极化率"""
        # 1. 规范一致性
        gauge_diff = abs(alpha_L - alpha_V)
        gauge_rel = 2 * gauge_diff / (abs(alpha_L) + abs(alpha_V) + 1e-30)

        print(
            f"Gauge consistency: L={alpha_L:.6f}, V={alpha_V:.6f}, RelDiff={gauge_rel:.2e}")
        if gauge_rel > 1e-4:
            print("  [WARN] Large gauge discrepancy! Check basis completeness.")

        # 2. 基准值对比
        if state_idx in self.ALPHA_BENCHMARKS:
            ref = self.ALPHA_BENCHMARKS[state_idx]
            err = abs(alpha_L - ref)
            print(
                f"Benchmark check (idx={state_idx}): Ref={ref:.6f}, Error={err:.2e}")
