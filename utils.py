import tensorflow as tf
import tensorflow_addons as tfa
from sionna.constants import PI
from tensorflow.keras.layers import Layer
import numpy as np
import matplotlib.pyplot as plt
import sionna.rt as rt
import json
import os
from sionna.channel import cir_to_ofdm_channel
import sionna
sionna.config.xla_compat=True

granularity = 5
scat_keep_prob = 5e-2

class Continuous_scatter(Layer):
    def __init__(self, alpha_r, alpha_i, lambda_, dtype=tf.float32, i_mask=True): # alpha_r support list input
        super(Continuous_scatter, self).__init__()
        self.alpha_r = tf.cast(alpha_r,dtype=dtype)
        self.alpha_i = tf.cast(alpha_i,dtype=dtype)
        self.lambda_ = tf.cast(lambda_,dtype=dtype)
        self.dtype_ = dtype
        self.i_mask = i_mask
        
    def build(self, input_shape):
        self.w_i = (1 - self.lambda_) * (1 - self.i_mask)
        self.w_r = self.lambda_
        self.t_i = self.alpha_i ** 0.5 * (1.6988 * self.alpha_i ** 2 + 10.8438 * self.alpha_i)\
              / (self.alpha_i ** 2 + 6.2201 * self.alpha_i + 10.2415)
        self.t_r = self.alpha_r ** 0.5 * (1.6988 * self.alpha_r ** 2 + 10.8438 * self.alpha_r)\
              / (self.alpha_r ** 2 + 6.2201 * self.alpha_r + 10.2415)
        self.a_u_i = 2 * PI / self.alpha_i * (1 - tf.exp(-self.alpha_i))
        self.a_u_r = 2 * PI / self.alpha_r * (1 - tf.exp(-self.alpha_r))
        self.a_b_i = 2 * PI / self.alpha_i * tf.exp(-2 * self.alpha_i) * (tf.exp(self.alpha_i) - 1)
        self.a_b_r = 2 * PI / self.alpha_r * tf.exp(-2 * self.alpha_r) * (tf.exp(self.alpha_r) - 1)

    #@tf.function(jit_compile=True, reduce_retracing=True)
    def call(self, object_id, points, k_i, k_s, n_hat): #only tested on SISO case
        shape = k_i.shape[0:-1]
        object_id = tf.reshape(object_id, [-1])
        k_i = tf.reshape(k_i, (-1, 3))
        k_s = tf.reshape(k_s, (-1, 3))
        n_hat = tf.reshape(n_hat, (-1, 3))
        t_r = tf.gather(self.t_r, object_id)
        alpha_r = tf.gather(self.alpha_r, object_id)
        a_u_r = tf.gather(self.a_u_r, object_id)
        a_b_r = tf.gather(self.a_b_r, object_id)

        dot_k_i_n = tf.reduce_sum(tf.multiply(k_i, n_hat), axis=1)
        k_r = k_i - 2 * tf.multiply(n_hat, tf.expand_dims(dot_k_i_n, axis=1))
        cosbeta_i = tf.reduce_sum(tf.multiply(-k_i, n_hat), axis=1)
        cosbeta_r = tf.reduce_sum(tf.multiply(k_r, n_hat), axis=1)
        s_i = (tf.exp(self.t_i) * tf.exp(self.t_i * cosbeta_i) - 1) \
                / ((tf.exp(self.t_i) - 1) * (tf.exp(self.t_i * cosbeta_i) + 1))
        s_r = (tf.exp(t_r) * tf.exp(t_r * cosbeta_r) - 1) \
                / ((tf.exp(t_r) - 1) * (tf.exp(t_r * cosbeta_r) + 1))
        a_i = self.a_u_i * s_i + self.a_b_i * (1 - s_i)
        a_r = a_u_r * s_r + a_b_r * (1 - s_r)
        cosbeta_i = tf.reduce_sum(tf.multiply(-k_i, k_s), axis=1)
        cosbeta_r = tf.reduce_sum(tf.multiply(k_r, k_s), axis=1)
        f_s = self.w_i / a_i * tf.exp(self.alpha_i * (cosbeta_i-1)) + self.w_r / a_r * tf.exp(alpha_r * (cosbeta_r-1))
        if shape != []:
            f_s = tf.reshape(f_s, shape)
        return f_s
    
    def visualize(self):
        plt.figure()
        for ii in range(len(self.alpha_r)):
            temp = plt.subplot(len(self.alpha_r) // 3 + 1, 3, ii + 1, projection='polar')
            
            theta_i = PI * 0.75
            k_i = -1 * tf.convert_to_tensor([tf.cos(theta_i), 0, tf.sin(theta_i)], dtype=self.dtype_)
            theta_s = np.linspace(0, PI, 100)
            n_hat = tf.convert_to_tensor([0, 0, 1], dtype=self.dtype_)
            f_s = []
            self.build(None)
            for jj in range(len(theta_s)):
                k_s = tf.convert_to_tensor([tf.cos(theta_s[jj]), 0, tf.sin(theta_s[jj])], dtype=self.dtype_)
                f_s.append(self.call(ii, None, k_i, k_s, n_hat))
            f_s = np.array(f_s)
            plt.polar(theta_s, np.abs(f_s), color='C1') 

def init_scene():
    scene_path = "sim_city"
    scene = rt.load_scene("/home/zhl/trial_project/scene/" + scene_path + "/" + scene_path + ".xml")
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", 
                                            polarization="V")
    scene.rx_array = scene.tx_array

    # config for sim_city
    Tx_pos = [0, 0, 5]
    scene.add(rt.Transmitter(name="Tx", position=Tx_pos))
    scene.add(rt.Receiver(name="Rx", position=[0, 0, 0]))
    for _, obj in scene.objects.items():
            obj.radio_material.relative_permittivity = 10
            obj.radio_material.conductivity = 0.1
            obj.radio_material.scattering_coefficient = 1
    return scene

def get_ind(paths):
    theta_r = tf.reshape(paths.theta_r, [-1])
    phi_r = tf.reshape(paths.phi_r, [-1])
    theta_start = np.min(theta_r) - 0.1
    theta_end = np.max(theta_r) + 0.1
    phi_start = np.max(phi_r) + 0.1
    phi_end = np.min(phi_r) - 0.1
    y = (theta_r - theta_start) / (theta_end - theta_start) * granularity
    y = tf.cast(y, dtype=tf.int32)
    x = (phi_r - phi_start) / (phi_end - phi_start) * granularity
    x = tf.cast(x, dtype=tf.int32)
    temp = tf.ones([granularity, len(theta_r)], dtype=tf.int32)
    eye = tf.linalg.diag(tf.linspace(0,granularity-1,granularity))
    temp = tf.matmul(tf.cast(eye, dtype=tf.int32), temp)
    i_aug = (x == temp)
    j_aug = (y == temp)
    i_aug = tf.split(i_aug, num_or_size_splits=granularity, axis=0)
    ind = tf.logical_and(j_aug, i_aug) #size:granularity^2*len(a)*datasize
    return ind

def loss_function(alpha_r, dataset, scene):
    # Rx_pos, traced_paths and ground_truth should share the same length
    scene.scattering_pattern_callable = Continuous_scatter(alpha_r=alpha_r, alpha_i=1, lambda_= 0.75)
    temp_loss = 0
    for ii in range(len(dataset)):
        Rx_pos, ground_truth, traced_paths = dataset[ii]
        paths = scene.compute_fields(*traced_paths, check_scene=False, scat_random_phases=False)
        amp = tf.reshape(paths.a, [-1])
        ind = get_ind(paths)
        a = tf.abs(amp)
        a = tf.math.pow(a, 2)
        ind = tf.cast(ind, dtype=tf.float32)
        heat_tab = tf.tensordot(ind, a, axes=1).numpy()
        heat_tab = np.log(heat_tab / np.sum(heat_tab) + 1e-30)
        
        temp_loss += tf.reduce_mean(tf.abs(heat_tab - ground_truth)).numpy()
    return temp_loss

def generate_dataset(Rx_pos_list, scene, alpha_r=[2,2,2,2,2]):
    dataset = []
    bandwidth = 50e6
    carrier_freq = 3.5e9
    N = 100
    frequencies = tf.linspace(carrier_freq-bandwidth/2, carrier_freq-bandwidth/2, N)
    amp = []
    tau = []
    scene.scattering_pattern_callable = Continuous_scatter(alpha_r=alpha_r, alpha_i=1, lambda_= 0.75)
    if not os.path.exists('/home/zhl/trial_project/data/' + str(alpha_r)):
        os.makedirs('/home/zhl/trial_project/data/' + str(alpha_r) )
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
    
    for Rx_pos in Rx_pos_list:
        scene.receivers["Rx"].position = Rx_pos
        traced_paths = scene.trace_paths(check_scene=False, 
                                        num_samples=1e6, 
                                        los=False, 
                                        reflection=False, 
                                        scattering=True, 
                                        scat_keep_prob=scat_keep_prob)
        paths = scene.compute_fields(*traced_paths, check_scene=False, scat_random_phases=True)
        amp = tf.reshape(paths.a, [-1])
        tau = tf.reshape(paths.tau, [-1])
        ind = get_ind(paths)
        ind = tf.reshape(ind, [granularity**2, len(amp)])
        heat_tab = func(ind)
        heat_tab = tf.reshape(heat_tab, [granularity, granularity]).numpy()
        plt.imshow(heat_tab)
        plt.savefig('/home/zhl/trial_project/data/' + str(alpha_r) + '/' + str(Rx_pos) + '.png')
        heat_tab = np.log(heat_tab / np.sum(heat_tab) + 1e-30)
        dataset.append([Rx_pos, heat_tab, traced_paths])
    return dataset