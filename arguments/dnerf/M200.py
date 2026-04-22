_base_ = './dnerf_default.py'

ModelHiddenParams = dict(
    kplanes_config = {
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 32,
        'resolution': [64, 64, 64, 64],
    }
)

OptimizationParams = dict(
    iterations = 14000,
    coarse_iterations = 3000,
    pruning_interval = 8000,
    percent_dense = 0.01,
)
