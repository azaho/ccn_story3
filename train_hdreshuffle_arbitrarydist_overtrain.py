import argparse, json, time, random, datetime
import hashlib, torch, math, pathlib, shutil, sys
import numpy as np
from torch import nn
from task_and_training_template import *

# PARSER START
parser = argparse.ArgumentParser(description='Train networks')
parser.add_argument('--net_size', type=int, help='size of input layer and recurrent layer', default=100)
parser.add_argument('--random', type=str, help='human-readable string used for random initialization', default="AA")
parser.add_argument('--shuffle_amount', type=float, help='how much to shift tunings?', default=130)
args = parser.parse_args()
# PARSER END

verbose = True  # print info in console?

hyperparameters.update({
    "random_string": str(args.random),  # human-readable string used for random initialization (for reproducibility)

    "train_for_steps": 20000,
})
task_parameters.update({
    "task_name": "2DIR1O",
    "input_direction_units": args.net_size,  # how many direction-selective input units?
    "dim_input": args.net_size + 1,  # plus one input for go cue signal
})
model_parameters.update({
    "model_name": "hdreshuffle_adCTRNN",
    "dim_recurrent": args.net_size,
    "dim_input": args.net_size + 1,  # plus one input for go cue signal
    "shuffle_amount": args.shuffle_amount
})
additional_comments += [
    "Reshuffle of tuning network, training is on top-level parameters + output layer"
]

#%FINAL_PARAMETERS_HERE%

directory = update_directory_name()
update_random_seed()

R1_i = torch.arange(model_parameters["dim_recurrent"])
R1_pref = R1_i/model_parameters["dim_recurrent"]*360
init_R1_pref_changes = torch.zeros(net_size)

# Modification of the class Model -- constrain the architecture to this particular solution class
class Model(Model):
    def __init__(self):
        super().__init__()

        del self.fc_x2ah, self.fc_h2ah
        self.b_ah = torch.zeros(self.dim_recurrent)
        self.b_y = torch.zeros(self.dim_output)

        self.R1_i = R1_i
        self.R1_pref = R1_pref
        self.IN_pref = torch.arange(task_parameters["input_direction_units"])/task_parameters["input_direction_units"]*360

        # TRAINABLE PARAMETERS:
        # 1: R1->R1 and input->R1 curve magnitudes
        # 2: R1 bias
        # Additionally: all the preference changes of R1 units
        self.top_parameters = nn.Parameter(torch.tensor([2, -1])/args.net_size*10)
        self.R1_pref_changes = nn.Parameter(init_R1_pref_changes/1800)#init_R1_pref_changes/180

    # output y and recurrent unit activations for all trial timesteps
    # input has shape (batch_size, total_time, dim_input) or (total_time, dim_input)
    # noise has shape (batch_size, total_time, dim_recurrent) or (total_time, dim_recurrent)
    def forward(self, input, noise):
        # build matrices based on top-level parameters
        self.W_h_ah = legi(self.R1_pref.repeat(len(self.R1_pref), 1), self.R1_pref.repeat(len(self.R1_pref), 1).T) * self.top_parameters[0]
        self.W_x_ah = legi((self.R1_pref+self.R1_pref_changes*1800).repeat(task_parameters["input_direction_units"], 1).T, self.IN_pref.repeat(len(R1_pref), 1)) * self.top_parameters[0]
        self.W_x_ah = torch.cat((self.W_x_ah, torch.zeros(len(R1_pref)).unsqueeze(1)), 1) # go cue has zero weights
        self.b_ah = torch.ones_like(self.b_ah) * self.top_parameters[1]

        if len(input.shape) == 2:
            # if input has size (total_time, dim_input) (if there is only a single trial), add a singleton dimension
            input = input[None, :, :]  # (batch_size, total_time, dim_input)
            noise = noise[None, :, :]  # (batch_size, total_time, dim_recurrent)
        batch_size, total_time, dim_input = input.shape
        ah = self.ah0.repeat(batch_size, 1)
        h = self.f(ah)
        hstore = []  # store all recurrent activations at all timesteps. Shape (batch_size, total_time, dim_recurrent)
        for t in range(total_time):
            ah = ah + (self.dt / self.tau) * (-ah + (self.W_h_ah@h.T).T + (self.W_x_ah@input[:, t].T).T + self.b_ah)
            h = self.f(ah) + noise[:, t, :]
            hstore.append(h)
        hstore = torch.stack(hstore, dim=1)
        output = self.fc_h2y(hstore)
        return output, hstore


if __name__ == "__main__":
    # train the network and save weights
    task = Task()
    model = Model()

    directory = update_directory_name()
    result = train_network(model, task, directory)

    save_metadata(directory, task, model, result)
    save_training_data(directory, result)
    model.save_firing_rates(task, "data_npy/" + directory[5:-1] + ".npy")
    save_metadata(directory, task, model, result, path="data_npy/" + directory[5:-1] + ".json")
    save_analysis_notebooks(directory, args)