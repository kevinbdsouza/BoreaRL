from typing import Dict
import numpy as np


def safe_update(T_old: float, dT: float, p: Dict) -> float:
    dT = np.clip(dT, -p['DT_CLIP'], p['DT_CLIP'])
    return np.clip(T_old + dT, p['T_MIN'], p['T_MAX'])


