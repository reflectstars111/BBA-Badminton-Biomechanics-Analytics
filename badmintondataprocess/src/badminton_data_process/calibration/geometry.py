from __future__ import annotations

from typing import Sequence


def project_point(
    homography: Sequence[Sequence[float]],
    point: tuple[float, float],
) -> tuple[float, float]:
    x, y = point
    h00, h01, h02 = homography[0]
    h10, h11, h12 = homography[1]
    h20, h21, h22 = homography[2]
    denom = h20 * x + h21 * y + h22
    if denom == 0:
        raise ZeroDivisionError("Homography projection denominator is zero")
    return (
        (h00 * x + h01 * y + h02) / denom,
        (h10 * x + h11 * y + h12) / denom,
    )

