import torch
from offlinerl.utils.exp import select_free_cuda
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
task = "Hopper-v3"
task_data_type = "low"
task_train_num = 99

seed = 42 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
obs_shape = None
act_shape = None
max_action = None

hidden_layer_size = 256
hidden_layers = 3
transition_layers = 4

transition_init_num = 7
transition_select_num = 5

real_data_ratio = 0.05

transition_batch_size = 256
policy_batch_size = 256
data_collection_per_epoch = 1e3
buffer_size = 1.2e6
steps_per_epoch = 100
max_epoch = 200

learnable_alpha = True
uncertainty_mode = 'aleatoric'
transition_lr = 1e-4
actor_lr = 3e-4
critic_lr = 3e-4
target_entropy = -3
discount = 0.99
soft_target_tau = 5e-3

horizon = 1
lam = 2

#tune
params_tune = {
    "buffer_size" : {"type" : "discrete", "value": [1e6, 2e6]},
    "real_data_ratio" : {"type" : "discrete", "value": [0.05, 0.1, 0.2]},
    "horzion" : {"type" : "discrete", "value": [1, 2, 5]},
    "lam" : {"type" : "continuous", "value": [0.1, 10]},
    "learnable_alpha" : {"type" : "discrete", "value": [True, False]},
}

#tune
grid_tune = {
    "horizon" : [1, 5],
    "lam" : [0.5, 1, 2, 5],
    "uncertainty_mode" : ['aleatoric', 'disagreement'],
}
