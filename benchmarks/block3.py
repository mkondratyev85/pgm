
settings = {
    'width' :  500 * 1_000,
    'height' : 500 * 1_000,
    'j_res' :  91,
    'i_res' : 91,
    'gx_0' : 0,
    'gy_0' : 10,
    'p0cell' : 0,
    'pdensity': 5,
    'MaxT' : '10 Myr',
    'plot_step_interval' : 0,
}

materials = [
    {'name': 'mobile wall', 'rho': 2520, 'eta': 10**16, 'mu': 8*10**10, 'C': 10, 'sinphi': 0},
    {'name': 'sand',        'rho': 1560, 'eta': 10**10, 'mu': 10**6, 'C': 10, 'sinphi': 0.58778525229247314},
]

boundaries = {
    'left_bound' : 'sleep',
    'top_bound' : 'sleep',
    'right_bound' : 'sleep',
    'bottom_bound' : 'sleep',
}

moving_cells = [
]
