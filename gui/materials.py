import numpy as np

materials = {
    "default": {"mu": 1, "rho": 1, "eta": 1, "C": 1, "sinphi": 1},
    "magma": {"mu": 8 * 10**10, "rho": 2800, "eta": 10**16, "C": 10**7, "sinphi": 45},
    "light magma": {"mu": 8 * 10**10, "rho": 2600, "eta": 10**13, "C": 10**7, "sinphi": 45},
    "heavy magma": {"mu": 8 * 10**10, "rho": 3200, "eta": 10**16, "C": 10**7, "sinphi": 45},
    "sand": {"mu": 10**6, "rho": 1560, "eta": 10**9, "C": 10, "sinphi": np.sin(np.radians(36))},
    "viso-elastic slab": {"mu": 10**10, "rho": 4000, "eta": 10**27, "C": 10, "sinphi": np.sin(np.radians(36))},
    "viso-elastic medium": {"mu": 10**20, "rho": 1, "eta": 10**24, "C": 10, "sinphi": np.sin(np.radians(36))},
    "sticky air": {"mu": 10**6, "rho": 1, "eta": 10**2, "C": 10, "sinphi": 0},
    "mobile wall": {"mu": 10**16, "rho": 2520, "eta": 10**22, "C": 10, "sinphi": 0},
}
