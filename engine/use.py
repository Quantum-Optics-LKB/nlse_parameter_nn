#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import torch
import numpy as np
from scipy.ndimage import zoom
from skimage.feature import hog
from engine.utils import normalize_data
from engine.generate import data_creation
from engine.model import Inception_ResNetv2
from engine.utils import plot_results, set_seed
set_seed(10)

def get_parameters(
        exp_path: str,
        saving_path: str, 
        resolution_out: int, 
        nlse_settings: tuple, 
        device_number: int, 
        cameras: tuple, 
        plot_generate_compare: bool
        ) -> tuple:
    """
    Perform parameter computation and visualization based on experimental data and model predictions.

    Parameters:
    - exp_path (str): Path to the experimental data.
    - saving_path (str): Directory where model checkpoints and results will be saved.
    - resolution_out (int): Output resolution for the computed parameters.
    - nlse_settings (tuple): Tuple containing NLSE-related settings: (n2, in_power, alpha, isat, waist, nl_length, delta_z, length).
    - device_number (int): GPU device number for computation.
    - cameras (tuple): Tuple containing camera settings: (_, _, _, resolution_training).
    - plot_generate_compare (bool): Flag indicating whether to plot comparison results.

    Returns:
    - tuple: Tuple containing computed parameters (computed_n2, computed_isat, computed_alpha).

    Performs the following steps:
    1. Initializes the neural network model (Inception_ResNetv2) and loads its pretrained weights.
    2. Loads experimental data from exp_path and preprocesses it (normalization, phase unwrapping, etc.).
    3. Computes predictions (outputs_n2, outputs_isat, outputs_alpha) using the loaded model.
    4. Computes actual physical parameters (computed_n2, computed_isat, computed_alpha) from the model outputs.
    5. Optionally, generates and plots comparison results between computed and experimental data.

    Note:
    - Assumes the model's state_dict is saved at saving_path based on resolution_out and nlse_settings.
    - Assumes exp_path contains experimental data that can be loaded using np.load().
    - Assumes data_creation() and plot_results() functions are defined elsewhere for data generation and visualization.
    """
    n2, in_power, alpha, isat, waist, nl_length, delta_z, length = nlse_settings
    _, _, _, resolution_training = cameras
    
    number_of_n2 = len(n2)
    number_of_isat = len(isat)
    number_of_alpha = len(alpha)

    min_n2 = n2.min()
    max_isat = isat.max()
    max_alpha = alpha.max()

    device = torch.device(f"cuda:{device_number}")
    cnn = Inception_ResNetv2(in_channels=4)
    cnn.to(device)

    directory_path = f'{saving_path}/training_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{in_power:.2f}/'
    directory_path += f'n2_net_w{resolution_out}_n2{number_of_n2}_isat{number_of_isat}_alpha{number_of_alpha}_power{in_power:.2f}.pth'
    cnn.load_state_dict(torch.load(directory_path))
    
    field = np.load(exp_path)

    density_experiment = normalize_data(zoom(np.abs(field), 
                (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
    phase_experiment = np.angle(field)
    phase_experiment = normalize_data(zoom(phase_experiment, 
                (resolution_training/field.shape[-2], resolution_training/field.shape[-1]))).astype(np.float16)
    
    fd, hog_phase_experiment = hog(phase_experiment[np.newaxis,:,:], orientations=8, pixels_per_cell=(6, 6), 
                          cells_per_block=(2, 2), visualize=True, channel_axis=0)

    fd, hog_density_experiment = hog(density_experiment[np.newaxis,:,:], orientations=8, pixels_per_cell=(6, 6), 
                        cells_per_block=(2, 2), visualize=True, channel_axis=0)

    
    E = np.zeros((1, 4, 256, 256), dtype=np.float16)
    E[0, 0, :, :] = density_experiment
    E[0, 1, :, :] = hog_density_experiment.astype(np.float16)
    E[0, 2, :, :] = phase_experiment
    E[0, 3, :, :] = hog_phase_experiment.astype(np.float16)
    
    with torch.no_grad():
        images = torch.from_numpy(E).float().to(device)
        outputs_n2, outputs_isat, outputs_alpha = cnn(images)
    
    computed_n2 = outputs_n2[0,0].cpu().numpy()*min_n2
    computed_isat = outputs_isat[0,0].cpu().numpy()*max_isat
    computed_alpha = outputs_alpha[0,0].cpu().numpy()*max_alpha

    print(f"n2 = {computed_n2} m^2/W")
    print(f"Isat = {computed_isat} W/m^2")
    print(f"alpha = {computed_alpha} m^-1")

    if plot_generate_compare:

        numbers = np.array([computed_n2]), in_power, np.array([computed_alpha]), np.array([computed_isat]), waist, nl_length, delta_z, length
        E = data_creation(numbers, cameras, device_number)
        plot_results(E, density_experiment, hog_density_experiment, phase_experiment, hog_phase_experiment, numbers, cameras, number_of_n2, number_of_isat, number_of_alpha, saving_path)
    
    return computed_n2, computed_isat, computed_alpha