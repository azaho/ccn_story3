import argparse, json, time, random, datetime
import hashlib, torch, math, pathlib, shutil, sys
import numpy as np
from torch import nn
from task_and_training_template import *

# PARSER START
parser = argparse.ArgumentParser(description='Train networks')
parser.add_argument('--net_size', type=int, help='size of input layer and recurrent layer', default=100)
parser.add_argument('--random', type=str, help='human-readable string used for random initialization', default="AA")
parser.add_argument('--la', type=float, help='L2 regularization coefficient', default=1e-4)
args = parser.parse_args()
# PARSER END

verbose = True  # print info in console?

hyperparameters.update({
    "random_string": str(args.random),  # human-readable string used for random initialization (for reproducibility)
    "regularization": "L2",  # options: L1, L2, None
    "regularization_lambda": args.la,

    "train_for_steps": 500,
    "save_network_every_steps": 1000,
})
task_parameters.update({
    "task_name": "2DIR1O",
    "input_direction_units": args.net_size,  # how many direction-selective input units?
    "dim_input": args.net_size + 1,  # plus one input for go cue signal
})
model_parameters.update({
    "model_name": "backpropndCTRNN",
    "dim_recurrent": args.net_size,
    "dim_input": args.net_size + 1,  # plus one input for go cue signal
})
additional_comments += [
]

#%FINAL_PARAMETERS_HERE%

directory = update_directory_name()
update_random_seed()

if __name__ == "__main__":
    # train the network and save weights
    model = Model()

    task_parameters["delay1_from"] = 10
    task_parameters["delay1_to"] = 50
    task_parameters["delay2_from"] = 70
    task_parameters["delay2_to"] = 90
    hyperparameters["train_for_steps"] = 2000
    task_parameters["distractor_visible"] = False
    directory = update_directory_name()
    task = Task()
    result = train_network(model, task, directory)

    print("===== SWITCHING =====")
    task_parameters["delay1_from"] = 10
    task_parameters["delay1_to"] = 120
    task_parameters["delay2_from"] = 140
    task_parameters["delay2_to"] = 160
    hyperparameters["train_for_steps"] = 10000
    task_parameters["distractor_visible"] = True
    directory = update_directory_name()
    task = Task()
    result = train_network(model, task, directory)

    save_metadata(directory, task, model, result)
    save_training_data(directory, result)
    model.save_firing_rates(task, "data_npy/" + directory[5:-1] + ".npy", noise_amplitude=hyperparameters["noise_amplitude"])
    save_metadata(directory, task, model, result, path="data_npy/" + directory[5:-1] + ".json")
    save_analysis_notebooks(directory, args)