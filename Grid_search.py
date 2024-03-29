import os # Configure which GPU 
gpu_num = 1 # Use "" to use the CPU
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
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e) 
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

tf.random.set_seed(1) # Set global random seed for reproducibility
            
import matplotlib.pyplot as plt
import numpy as np
import sionna.rt as rt
from sionna.constants import PI
from sionna.channel import cir_to_ofdm_channel
import pickle
import copy

from utils import *


with open('/home/zhl/trial_project/data/heat_tab1.pkl', 'rb') as file:
    heat_tab1 = pickle.load(file)

""" with open('/home/zhl/trial_project/data/heat_tab_low_res.pkl', 'rb') as file:
    heat_tab1 = pickle.load(file)
heat_tab1 = tfa.image.gaussian_filter2d(image=heat_tab1, filter_shape=(3,3), sigma=2.5, padding="CONSTANT", constant_values=0)
heat_tab1 /= np.sum(heat_tab1) """

alpha_r = [1,1,1,1,1]
# high_res:
# MSE
# converge at [7, 6, 19, 4, 5] after 26 iters, start from [1, 1, 1, 1, 1], final loss=6.2269083e-09
# converge at [6, 5, 19, 5, 5] after 2 iters, start from GT, final loss=6.013476e-09
# [19, 6, 10, 25, 3] -> [9, 5, 14, 1, 4]
# MAE
# converge at [6, 5, 20, 5, 5] after 35 iters, start from [1, 1, 1, 1, 1], final loss=1.6833572e-05
# converge at [6, 5, 20, 5, 5] after 3 iters, start from GT, final loss=1.6833572e-05
# [19, 6, 10, 25, 3] -> [8, 6, 20, 3, 4]
# low_res:
# MSE
# converge at [4, 6, 12, 1, 7] after 22 iters, start from [1, 1, 1, 1, 1], final loss=1.4388069e-05
# converge at [4, 6, 12, 1, 7] after 9 iters, start from GT, final loss=1.4388069e-05
# MAE
# converge at [7, 3, 21, 7, 12] after 28 iters, start from [1, 1, 1, 1, 1], final loss=0.0010024637
# converge at [7, 3, 21, 7, 12] after 13 iters, start from GT, final loss=0.0010024637
search_time = 0
truncate_counter = 0
alpha_i = 10
lambda_ = 0.75
scene_path = "sim_city"
scene = rt.load_scene("/home/zhl/trial_project/scene/" + scene_path + "/" + scene_path + ".xml")
scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", 
                                        polarization="V")
scene.rx_array = scene.tx_array

# config for sim_city
Tx_height = 5
Rx_pos_x = 10
Rx_pos_y = -10
Tx_pos = [0, 0, Tx_height]
Rx_pos = [Rx_pos_x, Rx_pos_y, 10]
scene.add(rt.Transmitter(name="Tx", position=Tx_pos))
scene.add(rt.Receiver(name="Rx", position=Rx_pos))
traced_paths = scene.trace_paths(check_scene=False, num_samples=1e6, los=False, reflection=False, scattering=True, scat_keep_prob=1)

while True:
    old_alpha_r = copy.deepcopy(alpha_r)
    mse = []
    early_stop_counter = 0
    for jj in range(40):
        alpha_r[search_time%5] = jj + 1e-2
        
        i = 0
        for _, obj in scene.objects.items():
            obj.radio_material.relative_permittivity = 10
            obj.radio_material.conductivity = 0.1
            obj.radio_material.scattering_coefficient =1
            obj.radio_material.scattering_pattern = rt.BackscatteringPattern(alpha_r=alpha_r[i], alpha_i=alpha_i, lambda_= lambda_)
            i = i + 1
        
        paths = scene.compute_fields(*traced_paths, check_scene=False, scat_random_phases=False)
        # Apply Monte Carlo on the scattering paths makes iteration faster, but hard to converge
        amp = tf.reshape(paths.a, [-1])
        tau = tf.reshape(paths.tau, [-1])
        theta_r = tf.reshape(paths.theta_r, [-1])
        phi_r = tf.reshape(paths.phi_r, [-1])
        #scene.preview(paths=paths)

        granularity = 108
        #print(f"resolution:{granularity}*{granularity}")
        theta_start = np.min(theta_r) - 0.1
        theta_end = np.max(theta_r) + 0.1
        phi_start = np.max(phi_r) + 0.1
        phi_end = np.min(phi_r) - 0.1
        bandwidth = 50e6
        carrier_freq = 3.5e9
        N = 100
        frequencies = tf.linspace(carrier_freq-bandwidth/2, carrier_freq-bandwidth/2, N)
        #a = tf.math.divide(a, tf.math.reduce_sum(a)) #normalize total energy
        y = (theta_r - theta_start) / (theta_end - theta_start) * granularity
        y = tf.cast(y, dtype=tf.int32)
        x = (phi_r - phi_start) / (phi_end - phi_start) * granularity
        x = tf.cast(x, dtype=tf.int32)
        temp = tf.ones([granularity, len(amp)], dtype=tf.int32)
        eye = tf.linalg.diag(tf.linspace(0,granularity-1,granularity))
        temp = tf.matmul(tf.cast(eye, dtype=tf.int32), temp)
        i_aug = (x == temp)
        j_aug = (y == temp)
        i_aug = tf.split(i_aug, num_or_size_splits=granularity, axis=0)
        ind = tf.logical_and(j_aug, i_aug) #size:granularity^2*len(a)*datasize
        # To consider the effect of bandwidth, run part1; or run part2
        # part1
        """ ind = tf.reshape(ind, [granularity**2, len(amp)])
        def cal_energy_in_band(ind):
            indices = tf.where(ind)
            tmp_a = tf.gather(amp, indices)
            tmp_a = tf.reshape(tmp_a, [1,1,1,1,1,len(tmp_a),1])
            tmp_tau = tf.gather(tau, indices)
            tmp_tau = tf.reshape(tmp_tau, [1,1,1,1,1,len(tmp_tau)])
            h_f = cir_to_ofdm_channel(frequencies=frequencies, a=tmp_a, tau=tmp_tau)
            h_f = tf.reshape(h_f, [-1])
            return tf.reduce_sum(tf.math.pow(tf.math.abs(h_f), 2)) / N
        @tf.function
        def func(elems):
            return tf.map_fn(cal_energy_in_band, elems, dtype=tf.float32)
        heat_tab = func(ind)
        heat_tab = tf.reshape(heat_tab, [granularity, granularity]).numpy() """
        # part2
        a = tf.abs(amp)
        a = tf.math.pow(a, 2)
        ind = tf.cast(ind, dtype=tf.float32)
        heat_tab = tf.tensordot(ind, a, axes=1).numpy()
        heat_tab = tfa.image.gaussian_filter2d(image=heat_tab, filter_shape=(5,5), sigma=10.0, padding="CONSTANT", constant_values=0)
        heat_tab /= np.sum(heat_tab)

        mse.append(tf.reduce_mean(tf.abs(heat_tab - heat_tab1)).numpy())
        print(mse[-1])
        if len(mse) >= 2:
            if mse[-1] >= mse[-2]:
                early_stop_counter += 1
            else:
                early_stop_counter = 0
        if early_stop_counter >= 5:
            print("Early stop...")
            break
    alpha_r[search_time%5] = mse.index(min(mse)) + 1
    print(alpha_r)
    search_time += 1
    if alpha_r == old_alpha_r:
        truncate_counter += 1
    else:
        truncate_counter = 0
    if truncate_counter >= 5:
        break

print(f"final alpha_r:{alpha_r}, search for {search_time} times")
#plt.plot(np.linspace(1,40,40), mse)