#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
import torch
import numpy as np
from scipy.ndimage import zoom
from skimage.restoration import unwrap_phase
from engine.nlse_generator import normalize_data

path = "/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN"
number_of_n2 = 10
number_of_power = 10
number_of_isat = 10
batch_size = 20
learning_rate = 0.001
power_labels = np.arange(0, number_of_power)
power_values = np.linspace(0.02, .5001, number_of_power)

resolution_out = 256

backend = "GPU"
if backend == "GPU":
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("---- DATA LOADING ----")

E = np.load("/home/louis/LEON/DATA/Atoms/2024/PINNS2/CNN/exp_data/field.npy")
E_data = np.zeros((E.shape[0], 3, E.shape[1], E.shape[2]))

cut = (E.shape[2] - E.shape[1])//2
E_reshape = E[:,0,:,cut:E.shape[2] - cut] 
E_resized =  zoom(E_reshape, (1, resolution_out/E_reshape.shape[1],resolution_out/E_reshape.shape[2]), order=3)

E = np.zeros((E.shape[0], 3, resolution_out, resolution_out))
E[:,0,:,:] = np.abs(E_resized)**2
E[:,1,:,:] = np.angle(E_resized)
E[:,2,:,:] = unwrap_phase(np.angle(E_resized))
E= normalize_data(E)


classes = {
        'n2': tuple(map(str, np.linspace(-1e-9, -1e-10, number_of_n2))),
        'power' : tuple(map(str, np.linspace(0.02, .5001, number_of_power))),
        'isat' : tuple(map(str, np.linspace(1e4, 1e6, number_of_isat) ))
    }

field_data = [E[:,[0],:,:], E[:,[0,1],:,:], E[:,[0,2],:,:], E[:,[1],:,:], E[:,[2],:,:], E]
data_types = ["amp", "amp_pha", "amp_pha_unwrap", "pha", "pha_unwrap", "amp_pha_pha_unwrap"]

data_types_results = {}
for data_types_index in range(len(data_types)):

    E = field_data[data_types_index]

    models = {}
    for model_index in range(2, 6):
            
        if model_index == 2:
            from multiple_power.model.model_resnetv2 import Inception_ResNetv2
        elif model_index == 3:
            from multiple_power.model.model_resnetv3 import Inception_ResNetv2
        elif model_index == 4:
            from multiple_power.model.model_resnetv4 import Inception_ResNetv2
        elif model_index == 5:
            from multiple_power.model.model_resnetv5 import Inception_ResNetv2
        
        model_version =  str(Inception_ResNetv2).split('.')[-2]
        
        stamp = f"multiple_power_{data_types[data_types_index]}_{model_version}"
        new_path = f"{path}/{stamp}_training" 

        
        cnn = Inception_ResNetv2(in_channels=E.shape[1], class_n2=number_of_n2, class_power=number_of_power, class_isat=number_of_isat)
        cnn = cnn.to(device)
        cnn.load_state_dict(torch.load(f'{new_path}/n2_net_w{resolution_out}_n2{number_of_n2}_isat{number_of_isat}_power{number_of_power}.pth'))

        with torch.no_grad():
            images = torch.from_numpy(E).float().to(device)
            labels_power = torch.from_numpy(power_labels).long().to(device)
            powers_values = torch.from_numpy(power_values[:,np.newaxis]).float().to(device)
                
            outputs_n2, outputs_power, outputs_isat = cnn(images, powers_values)
            _, predicted_n2 = torch.max(outputs_n2, 1)
            _, predicted_power = torch.max(outputs_power, 1)
            _, predicted_isat = torch.max(outputs_isat, 1)

            result_n2 = classes['n2'][predicted_n2]
            result_index_n2 = predicted_n2

            result_isat = classes['isat'][predicted_isat]
            result_index_isat = predicted_isat

            result_power = classes['power'][predicted_power]
            result_index_power = predicted_power
    
    models[model_version] = [result_n2, result_index_n2,result_power, result_index_power, result_isat, result_index_isat]
data_types_results[data_types[data_types_index]] = models


for data_types_index in range(len(data_types)):
    for model_index in range(2, 6):
        result_n2 = np.array(data_types_results[data_types[data_types_index]][f"model_resnetv{model_index}"][0])
        result_index_n2 = np.array(data_types_results[data_types[data_types_index]][f"model_resnetv{model_index}"][1])
        result_isat = np.array(data_types_results[data_types[data_types_index]][f"model_resnetv{model_index}"][2])
        result_index_isat = np.array(data_types_results[data_types[data_types_index]][f"model_resnetv{model_index}"][3])
        
        print(f"For {data_types[data_types_index]} and model_resnetv{model_index} :\n") 
        print(f"Average n2 = {np.mean(result_n2)}")
        print(f"Std n2 = {np.std(result_n2)}")

        print(f"Average index n2 = {np.mean(result_index_n2)}")
        print(f"Std index n2 = {np.std(result_index_n2)}")

        print(f"Average isat = {np.mean(result_isat)}")
        print(f"Std isat = {np.std(result_isat)}")

        print(f"Average index isat = {np.mean(result_index_isat)}")
        print(f"Std index isat = {np.std(result_index_isat)}")