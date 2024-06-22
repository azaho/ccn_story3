import argparse, json, time, random, datetime
import hashlib, torch, math, pathlib, shutil, sys
import numpy as np
from torch import nn
from task_and_training_template import *

# PARSER START
parser = argparse.ArgumentParser(description='Train networks')
parser.add_argument('--net_size', type=int, help='size of input layer and recurrent layer', default=120*2)
parser.add_argument('--random', type=str, help='human-readable string used for random initialization', default="AA")
parser.add_argument('--gate_proportion', type=float, help='proportion of gate units in the RNN', default=0.5)
args = parser.parse_args()
# PARSER END

verbose = True  # print info in console?

hyperparameters.update({
    "random_string": str(args.random),  # human-readable string used for random initialization (for reproducibility)
})
task_parameters.update({
    "task_name": "2DIR1O",
    "input_direction_units": args.net_size,  # how many direction-selective input units?
    "dim_input": args.net_size + 1,  # plus one input for go cue signal
})
model_parameters.update({
    "model_name": "hdgating_and_reshuffle_fMST2CTRNN",
    "dim_recurrent": args.net_size,
    "dim_input": args.net_size + 1,  # plus one input for go cue signal
})
additional_comments += [
    "Gating+Reshuffle network, training is on top-level parameters + output layer",
    "R1 pref changes are fixed to be equal to those of MST data"
]

#%FINAL_PARAMETERS_HERE%

directory = update_directory_name()
update_random_seed()

proportion_gate = args.gate_proportion
R1_si, R1_ei = 0, int(model_parameters["dim_recurrent"]*(1-proportion_gate))
R1_i = torch.arange(R1_si, R1_ei)
DT_si, DT_ei = R1_ei, model_parameters["dim_recurrent"]
DT_i = torch.arange(DT_si, DT_ei)

DT_pref = DT_i/len(DT_i)*360

# from MST data
R1_pref = [127.0, 296.6, 178.8, 359.1, 322.9, 53.5, 83.8, 231.9, 220.1, 151.3, 328.4, 266.7, 254.9, 49.3, 165.2, 159.6, 169.0, 150.7, 140.2, 304.0, 289.7, 358.1, 139.1, 330.4, 150.7, 347.5, 153.2, 354.7, 25.7, 132.5, 313.6, 335.9, 309.7, 327.1, 286.1, 328.3, 151.9, 151.6, 139.1, 135.0, 73.4, 305.4, 223.8, 286.1, 293.3, 35.6, 269.0, 349.4, 53.4, 268.2, 251.7, 267.5, 266.3, 258.1, 259.0, 338.6, 313.0, 315.7, 163.3, 123.1, 127.0, 296.6, 178.8, 359.1, 322.9, 53.5, 83.8, 231.9, 220.1, 151.3, 328.4, 266.7, 254.9, 49.3, 165.2, 159.6, 169.0, 150.7, 140.2, 304.0, 289.7, 358.1, 139.1, 330.4, 150.7, 347.5, 153.2, 354.7, 25.7, 132.5, 313.6, 335.9, 309.7, 327.1, 286.1, 328.3, 151.9, 151.6, 139.1, 135.0, 73.4, 305.4, 223.8, 286.1, 293.3, 35.6, 269.0, 349.4, 53.4, 268.2, 251.7, 267.5, 266.3, 258.1, 259.0, 338.6, 313.0, 315.7, 163.3, 123.1]
R1_pref_changes = [-23.2, 3.2, 104.6, 47.3, 32.5, -240.7, -221.4, 105.6, 66.2, 12.2, 174.2, 140.0, 168.3, -101.2, 43.3, 150.3, 22.7, -149.6, -6.7, 13.1, -36.9, 9.3, 18.6, 19.9, 18.5, 171.1, -6.7, 214.5, -118.4, -9.0, 16.5, 43.7, 5.2, 2.0, -18.0, 10.4, 8.6, 22.7, 16.0, -1.5, -83.7, 196.2, 85.7, 188.9, 200.2, -139.4, 142.2, 212.1, -116.0, 119.0, 137.9, 105.4, 149.2, 132.5, 181.7, 230.4, 184.7, 180.8, 3.3, 0.8, 23.2, -3.2, -104.6, -47.3, -32.5, 240.7, 221.4, -105.6, -66.2, -12.2, -174.2, -140.0, -168.3, 101.2, -43.3, -150.3, -22.7, 149.6, 6.7, -13.1, 36.9, -9.3, -18.6, -19.9, -18.5, -171.1, 6.7, -214.5, 118.4, 9.0, -16.5, -43.7, -5.2, -2.0, 18.0, -10.4, -8.6, -22.7, -16.0, 1.5, 83.7, -196.2, -85.7, -188.9, -200.2, 139.4, -142.2, -212.1, 116.0, -119.0, -137.9, -105.4, -149.2, -132.5, -181.7, -230.4, -184.7, -180.8, -3.3, -0.8]
# remove symmetrizing
#R1_pref = R1_pref[:65]
#R1_pref_changes = R1_pref_changes[:65]

R1_pref = np.arange(len(R1_pref))/len(R1_pref)*360
np.random.shuffle(R1_pref)
np.random.shuffle(R1_pref_changes)

R1_pref = torch.tensor(R1_pref)
R1_pref_changes = torch.tensor(R1_pref_changes)

# Modification of the class Model -- constrain the architecture to this particular solution class
class Model(Model):
    def __init__(self):
        super().__init__()

        del self.fc_x2ah, self.fc_h2ah
        self.b_ah = torch.zeros(self.dim_recurrent)
        self.b_y = torch.zeros(self.dim_output)

        self.R1_si, self.R1_ei = R1_si, R1_ei
        self.R1_i = R1_i
        self.DT_si, self.DT_ei = DT_si, DT_ei
        self.DT_i = DT_i
        self.R1_pref = R1_pref
        self.DT_pref = DT_pref
        self.IN_pref = torch.arange(task_parameters["input_direction_units"])/task_parameters["input_direction_units"]*360
        self.R1_pref_changes = R1_pref_changes#nn.Parameter(R1_pref_changes/1800)

        # TRAINABLE PARAMETERS:
        # 1: input-DT, DT-R1, R1-R1 curve magnitude
        # 2: R1 bias and DT bias
        # 3: R1-DT nonspecific connection magnitude
        self.top_parameters = nn.Parameter(torch.tensor([3, -0.5, -0.5])/args.net_size*10)

    # output y and recurrent unit activations for all trial timesteps
    # input has shape (batch_size, total_time, dim_input) or (total_time, dim_input)
    # noise has shape (batch_size, total_time, dim_recurrent) or (total_time, dim_recurrent)
    def forward(self, input, noise):
        # build matrices based on top-level parameters
        W_h_ah = torch.zeros((self.dim_recurrent, self.dim_recurrent), dtype=torch.float32)
        W_x_ah = torch.zeros((self.dim_recurrent, self.dim_input), dtype=torch.float32)
        b_ah = torch.zeros_like(self.b_ah)
        W_h_ah[R1_si:R1_ei, R1_si:R1_ei] = legi(self.R1_pref.repeat(len(self.R1_pref), 1), self.R1_pref.repeat(len(self.R1_pref), 1).T) * self.top_parameters[0]
        W_h_ah[R1_si:R1_ei, DT_si:DT_ei] = legi((self.R1_pref+self.R1_pref_changes).repeat(len(self.DT_pref), 1).T, self.DT_pref.repeat(len(self.R1_pref), 1)) * self.top_parameters[0]

        W_h_ah[DT_si:DT_ei, R1_si:R1_ei] = torch.ones((len(DT_i), len(R1_i))) * self.top_parameters[2]

        W_x_ah[DT_si:DT_ei, :-1] = legi(self.DT_pref.repeat(task_parameters["input_direction_units"], 1).T, self.IN_pref.repeat(len(self.DT_pref), 1)) * self.top_parameters[0]
        b_ah[R1_si:R1_ei] = torch.ones((len(R1_i),)) * self.top_parameters[1]
        b_ah[DT_si:DT_ei] = torch.ones((len(DT_i),)) * self.top_parameters[1]

        self.W_h_ah = W_h_ah
        self.W_x_ah = W_x_ah
        self.b_ah = b_ah
        #im = plt.imshow(W_h_ah.detach())
        #im = plt.imshow(legi(self.R1a_pref.repeat(len(self.R1a_pref), 1), self.R1a_pref.repeat(len(self.R1a_pref), 1).T) * self.top_parameters[0].detach())
        #plt.colorbar(im)
        #plt.show()
        #exit()

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