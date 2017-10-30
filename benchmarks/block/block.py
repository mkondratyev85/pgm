
settings = {
    'width' :  500 * 1_000,
    'height' : 500 * 1_000,
    'j_res' :  41,
    'i_res' : 41,
    'gx_0' : 0,
    'gy_0' : 10,
    'p0cell' : 0,
    'pdensity': 5,
    'MaxT' : '20 Myr',
    'plot_step_interval' : 0,
}

materials = [
    {'name':"heavy magma",  "rho": 3200, "eta": 10**12, "mu": 8 * 10**10, "C": 10**7, "sinphi": 45},
    {'name': 'sand',        'rho': 1500, 'eta': 10**12, 'mu': 10**6,      'C': 10, 'sinphi': 0.58778525229247314},
]

boundaries = {
    'left_bound' : 'sleep',
    'top_bound' : 'sleep',
    'right_bound' : 'sleep',
    'bottom_bound' : 'sleep',
}

moving_cells = [
]
