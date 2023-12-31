import argparse, json, time, random, datetime
import hashlib, torch, math, pathlib, shutil, sys
import numpy as np
from torch import nn
from task_and_training_template import *

# PARSER START
parser = argparse.ArgumentParser(description='Train networks')
parser.add_argument('--net_size', type=int, help='size of input layer and recurrent layer', default=net_size)
parser.add_argument('--random', type=str, help='human-readable string used for random initialization', default="AA")
parser.add_argument('--la', type=float, help='L2 regularization coefficient', default=0)
args = parser.parse_args()
# PARSER END

verbose = True  # print info in console?

hyperparameters.update({
    "random_string": str(args.random),  # human-readable string used for random initialization (for reproducibility)
    "regularization": "L2_weights",  # options: L1, L2, None
    "regularization_lambda": args.la,
})
task_parameters.update({
    "task_name": "2DIR1O",
    "input_direction_units": args.net_size,  # how many direction-selective input units?
    "dim_input": args.net_size + 1,  # plus one input for go cue signal
})
model_parameters.update({
    "model_name": "backprop2CTRNN",
    "dim_recurrent": args.net_size,
    "dim_input": args.net_size + 1,  # plus one input for go cue signal
})
additional_comments += [
    "Trained with Adam from a random initialization",
    "The first 10% of training time, the networks are trained on the task where the network has to output target AND distractor"
]

#%FINAL_PARAMETERS_HERE%

directory = update_directory_name()
update_random_seed()

if __name__ == "__main__":
    total_train_steps = hyperparameters["train_for_steps"]

    # train the network and save weights
    task_parameters.update(pretrain_delays)
    model_parameters["dim_output"] = 4
    task_parameters["dim_output"] = 4
    hyperparameters["train_for_steps"] = total_train_steps//10
    directory = update_directory_name()
    model_pretrain = Model()
    task = Task_outputO1O2()
    result = train_network(model_pretrain, task, directory)

    # train the network and save weights
    print("===== SWITCHING =====")
    task_parameters.update(final_delays)
    hyperparameters["train_for_steps"] = total_train_steps*9//10
    model_parameters["dim_output"] = 2
    task_parameters["dim_output"] = 2
    directory = update_directory_name()
    model = Model()
    with torch.no_grad():
        model.fc_h2ah.weight = torch.nn.Parameter(model_pretrain.fc_h2ah.weight)
        model.fc_h2y.weight = torch.nn.Parameter(model_pretrain.fc_h2y.weight[:2])
        model.fc_x2ah.weight = torch.nn.Parameter(model_pretrain.fc_x2ah.weight)
        model.fc_x2ah.bias = torch.nn.Parameter(model_pretrain.fc_x2ah.bias)
    del model_pretrain
    del task
    task = Task()
    result = train_network(model, task, directory)

    save_metadata(directory, task, model, result)
    save_training_data(directory, result)
    model.save_firing_rates(task, "data_npy/" + directory[5:-1] + ".npy")
    save_metadata(directory, task, model, result, path="data_npy/" + directory[5:-1] + ".json")
    save_analysis_notebooks(directory, args)
    output_connectivity_factors(directory, task, model)