import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def read_text_auto(path: str) -> str:
    # Read bytes, then decode based on BOM or fallback
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")          # BOM will set LE/BE correctly
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")       # UTF-8 with BOM
    # Fallback: UTF-8, but don’t crash on odd bytes
    return raw.decode("utf-8", errors="replace")

def load_gyro_triplets(path: str) -> np.ndarray:
    """
    Returns Nx3 array (wx, wy, wz) in deg/s from your log format.
    Assumes the 5th '|' field contains 'gx,gy,gz'.
    """
    text = read_text_auto(path)
    rows = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) > 5:
            gyro_field = parts[4]
            vals = gyro_field.split(",")
            if len(vals) >= 3:
                try:
                    # Convert to float and to deg/s if they were in rad/s
                    wx = float(vals[0]) * 57.2958
                    wy = float(vals[1]) * 57.2958
                    wz = float(vals[2]) * 57.2958
                    rows.append([wx, wy, wz])
                except ValueError:
                    # Skip lines with non-numeric gyro values
                    pass
    if not rows:
        raise ValueError("No gyro rows parsed — check column index or file format.")
    return np.asarray(rows, dtype=float)

def AllanDeviation(dataArr: np.ndarray, fs: float, maxNumM: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Allan deviation for equally spaced samples of 'angle' (or integrated rate).
    """
    ts = 1.0 / fs
    N = len(dataArr)
    if N < 10:
        raise ValueError("Not enough samples for Allan deviation.")

    # Valid M must satisfy N - 2M > 0 (need at least 2 clusters & one diff)
    Mmax = int(2 ** np.floor(np.log2(N / 2)))
    if Mmax < 1:
        raise ValueError("Window too small for Allan deviation.")

    M = np.logspace(np.log10(1), np.log10(Mmax), num=maxNumM)
    M = np.unique(np.ceil(M).astype(int))
    # Enforce constraint explicitly
    M = M[M < (N // 2)]
    if len(M) == 0:
        raise ValueError("No valid M values (N too small).")
    taus = M * ts

    allanVar = np.zeros(len(M), dtype=float)
    for i, mi in enumerate(M):
        twoMi = 2 * mi
        diff = (dataArr[twoMi:N] - 2.0 * dataArr[mi:N - mi] + dataArr[0:N - twoMi])
        allanVar[i] = np.sum(diff * diff)

    allanVar /= (2.0 * (taus ** 2) * (N - 2.0 * M))
    return taus, np.sqrt(allanVar)

# ----- use it -----
path = "/Users/dsong/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/Club Stuff/LRI/FV-Controls/Control/dataTest1072025.txt"
gyro = load_gyro_triplets(path)     # Nx3, deg/s

fs = 10.0
ts = 1.0 / fs

# If your log is rate (deg/s), integrating once gives angle (deg).
# The discrete Allan formula here is for the phase-like quantity built from cumulative sum.
ang = np.cumsum(gyro, axis=0) * ts  # angle (deg)

tausX, devX = AllanDeviation(ang[:, 0], fs, 1000)
tausY, devY = AllanDeviation(ang[:, 1], fs, 1000)
tausZ, devZ = AllanDeviation(ang[:, 2], fs, 1000)

# tau closest to 1 second
idx = int(np.argmin((tausX - 1.0) ** 2))

plt.plot(tausX, devX, label="Wx")
plt.plot(tausY, devY, label="Wy")
plt.plot(tausZ, devZ, label="Wz")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Tau (s)")
plt.ylabel("Allan Deviation")
plt.show()

print("Idx @ tau≈1s:", idx, "tau=", tausX[idx])
print("Allan dev @ 1s (Wx, Wy, Wz), deg:", devX[idx], devY[idx], devZ[idx])

# Your conversions:
# - Multiply by 60 converts deg/sqrt(s) → deg/√hr (common for ARW)
print("ARW estimate (deg/√hr) Wx, Wy, Wz:",
      devX[idx] * 60, devY[idx] * 60, devZ[idx] * 60)

# Bias instability readout (make sure this 'factor' matches your chosen convention)
factor = 5421.68674699
print("Bias instability proxy Wx, Wy, Wz:",
      devX[-1] * factor, devY[-1] * factor, devZ[-1] * factor)
