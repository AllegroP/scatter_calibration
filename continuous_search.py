import os
import time
# Configure which GPU 
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
import sionna
sionna.config.xla_compat=True #需要进一步改动代码，以加快执行速度

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
import tensorflow_addons as tfa
gpus = tf.config.list_physical_devices('GPU')


tf.random.set_seed(1) # Set global random seed for reproducibility
            
import numpy as np
import scipy.optimize as opt
import sionna.rt as rt
from sionna.constants import PI
from sionna.channel import cir_to_ofdm_channel
import pickle
import copy

import utils

alpha_r = np.array([1e-3,12,13,2,5])
# high_res:
# MAE: [1,1,1,1,1] -> [2.2, 2.6, 1.8, 3.4, 1.4], loss=1.9677054e-05 gt=[2, 2, 2, 2, 2]
# [1e-3,12,13,2,5] -> [3.  0.6 2.8 4.  2.4] loss=0.13210774958133698

# use 2 nd point:
# [1e-3,12,13,2,5] -> [2.2 2.8 2.4 2.4 2.6] loss=0.42228537797927856
# use 2 points:
# [1e-3,12,13,2,5] -> [2.6 2.8 2.4 2.6 2. ], loss=0.5580779314041138
# use 50 points:
# [1e-3,12,13,2,5] -> [2.2 2.6 2.6 2.  3.4], loss=31.62438040971756　res=10
# [1e-3,12,13,2,5] -> [2.6 2.4 2.2 2.  2.8] loss=40.31465381383896 res=5
Rx_pos_list = []
r = 5
for ii in range(50):
    theta = (1-ii/50)*np.pi/2 + ii/50*2*np.pi
    Rx_pos_list.append([r * np.cos(theta), r * np.sin(theta), 5])
scene = utils.init_scene()
dataset = utils.generate_dataset(Rx_pos_list, scene, alpha_r=[2,2,2,2,2])

search_time = 0
truncate_counter = 0
while True:
    old_alpha_r = copy.deepcopy(alpha_r)
    loss = []
    early_stop_counter = 0
    for jj in range(40):
        alpha_r[(search_time)%5] = (jj + 1) * 0.2
        loss.append(utils.loss_function(alpha_r, 
                                        dataset,
                                        scene=scene))
        print(loss[-1])
        if len(loss) >= 2:
            if loss[-1] >= loss[-2]:
                early_stop_counter += 1
            else:
                early_stop_counter = 0
        if early_stop_counter >= 5:
            print("Early stop...")
            break
    alpha_r[(search_time)%5] = (loss.index(min(loss)) + 1) * 0.2
    print(alpha_r)
    search_time += 1
    if np.array_equal(alpha_r, old_alpha_r):
        truncate_counter += 1
    else:
        truncate_counter = 0
    if truncate_counter >= 5:
        break

print(f"final alpha_r:{alpha_r}, search for {search_time} times")
print(utils.loss_function(alpha_r=[2,2,2,2,2], 
                          dataset=dataset, 
                          scene=scene))

# solution = opt.minimize(utils.loss_function, alpha_r, args=(ground_truth, traced_paths_list, scene), method='Powell',\
#                         bounds=[(0,40), (0,40), (0,40), (0,40), (0,40)]) 

# Nelder-Mead: [1,1,1,1,1]-> [ 1.847e+00  3.339e+00  1.210e+00  3.152e-05  8.586e-01] loss=2.3974851501407102e-05
# Powell: [1,1,1,1,1]->[ 2.274e+00  2.372e+00  2.023e+00  3.573e+00  1.593e+00] loss=1.9563888e-05
# [1e-3,12,13,2,5]->[ 2.275e+00  2.379e+00  2.004e+00  3.566e+00  1.587e+00] 1.9562913e-05
# TNC & L-BFGS-B: remain unchanged