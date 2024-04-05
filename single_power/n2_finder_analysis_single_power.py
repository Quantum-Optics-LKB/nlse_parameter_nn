#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Louis Rossignol
from tqdm import tqdm
import pandas as pd


def analysis(path, power_values):

    data_types = ["amp", "amp_pha", "amp_pha_unwrap", "pha", "pha_unwrap", "amp_pha_pha_unwrap"]
    data_type_results = {}
    for data_types_index in range(len(data_types)):

        models = {}
        for model_index in range(2, 6):
            
            model_version =  f"model_resnetv{model_index}_1powers"
            model = {}
            power_list_accuracy = []
            power_list_index_error = []
            for power in tqdm(power_values, position=4,desc="Iteration", leave=False):
                
                stamp = f"power{str(power)[:4]}_{data_types[data_types_index]}_{model_version}"
                new_path = f"{path}/{stamp}_training"

                f = open(f'{new_path}/testing.txt', 'r')

                count = 0
                for line in f.readlines():
                    if "TESTING" in line:
                        break
                    count += 1
                f.close()

                f = open(f'{new_path}/testing.txt', 'r')
                lines = f.readlines()
                accuracy = lines[count+2].split(" ")[-1].split("%")[0]
                power_list_accuracy.append(float(accuracy))
                if len(accuracy) == len("100.00"):
                    power_list_index_error.append(0)
                else:
                    power_list_index_error.append(float(lines[count+13].split(" ")[-1].split("\n")[0]))
                f.close
            model["accuracy"] = power_list_accuracy
            model["index_error"] = power_list_index_error 
                
        models[model_version] = model
    data_type_results[data_types[data_types_index]] = models

    df = pd.DataFrame.from_dict(data_type_results, orient="columns")
    df.to_json(f'{path}/model_analysis_single_power.json')            
                
                