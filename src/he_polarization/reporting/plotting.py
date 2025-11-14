"""绘制动力学极化率曲线，复现论文图 3.12-3.13 的频率扫描。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class DynamicPolarizabilityPlotter:
    """负责动力学极化率的可视化输出。"""

    def plot(self, freqs: Iterable[float], length: Iterable[float], velocity: Iterable[float], acceleration: Iterable[float], *, title: str, output_path: str | None = None) -> None:
        """绘制三种规范的极化率曲线，并可选保存。"""
        freqs_arr = np.asarray(tuple(freqs), dtype=float)
        length_arr = np.asarray(tuple(length), dtype=float)
        velocity_arr = np.asarray(tuple(velocity), dtype=float)
        acceleration_arr = np.asarray(tuple(acceleration), dtype=float)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(freqs_arr, length_arr, label="长度规范", lw=1.6)
        ax.plot(freqs_arr, velocity_arr, label="速度规范", lw=1.6)
        ax.plot(freqs_arr, acceleration_arr, label="加速度规范", lw=1.6)
        ax.set_xlabel("\u9891\u7387 / a.u.")
        ax.set_ylabel("\u52a8\u529b\u5b50\u6781\u5316\u7387 / a.u.")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_dynamic_polarizability(freqs: Iterable[float], length: Iterable[float], velocity: Iterable[float], acceleration: Iterable[float], *, title: str, output_path: str | None = None) -> None:
    """便捷函数，内部调用 :class:`DynamicPolarizabilityPlotter`。"""
    plotter = DynamicPolarizabilityPlotter()
    plotter.plot(freqs, length, velocity, acceleration,
                 title=title, output_path=output_path)
