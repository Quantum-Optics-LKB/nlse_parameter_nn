#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol

import numpy as np
from engine.utils import set_seed
set_seed(10)

def data_augmentation(
    E: np.ndarray,
    labels: tuple,
    ) -> np.ndarray:
    """
    Perform data augmentation on the provided dataset by adding fringes at random rotations, to the images.

    Args:
        E (np.ndarray): A 4D numpy array containing the image data with dimensions 
            (number_of_images, channels, height, width).
        labels (tuple): A tuple containing the number of each type of label and the 
            labels themselves. Expected to be in the format:
            (number_of_n2, n2_labels, number_of_isat, isat_labels, number_of_alpha, alpha_labels).

    Returns:
        np.ndarray: A 4D numpy array containing the augmented image data with dimensions 
            (augmented_number_of_images, channels, height, width).
        tuple: A tuple containing the augmented labels in the same format as the input labels.

    Description:
        This function augments the input image dataset by generating new images through 
        the following methods:
        - Fringes perturbations with fixed line counts (50 and 100) at different noises and angles.

        The augmentation process involves creating copies of the original images and 
        applying the aforementioned transformations to generate additional augmented images.

    Steps:
        1. Shuffle the original dataset and labels to ensure randomness.
        2. Repeat the labels to match the size of the augmented dataset.
        3. Return the augmented dataset and corresponding labels.
    """
    
    number_of_n2, n2_labels, number_of_isat, isat_labels, number_of_alpha, alpha_labels = labels

    augmentation = 16

    indices = np.arange(len(n2_labels))
    np.random.shuffle(indices)

    n2_labels = n2_labels[indices]
    isat_labels = isat_labels[indices]
    alpha_labels = alpha_labels[indices]
    E = E[indices, :, :, :]

    n2_labels = np.repeat(n2_labels, augmentation)
    isat_labels = np.repeat(isat_labels, augmentation)
    alpha_labels = np.repeat(alpha_labels, augmentation)
    augmented_data = np.repeat(E, augmentation, axis=0)
    
    labels = (number_of_n2, n2_labels, number_of_isat, isat_labels, number_of_alpha, alpha_labels)
              
    return augmented_data, labels