import argparse, json, time, random, datetime
import hashlib, torch, math, pathlib, shutil, sys
import numpy as np
from torch import nn
from task_and_training_template import *

# PARSER START
parser = argparse.ArgumentParser(description='Train networks')
parser.add_argument('--net_size', type=int, help='size of input layer and recurrent layer', default=130*2)
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
    "model_name": "hdgating_and_reshuffle_fMST1CTRNN",
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
R1_pref = [168.9, 334.9, 245.3, 321.5, 125.9, 272.8, 135.3, 80.7, 122.4, 328.6, 95.9, 309.2, 247.8, 311.1, 324.0, 281.9, 316.3, 184.4, 247.5, 270.9, 302.7, 280.1, 4.6, 275.0, 219.0, 220.3, 353.8, 220.5, 197.6, 176.6, 152.6, 91.8, 92.0, 113.5, 359.4, 205.3, 159.1, 247.1, 30.7, 357.7, 333.3, 351.8, 329.8, 50.5, 257.9, 153.0, 102.4, 236.9, 328.6, 311.5, 284.7, 349.7, 165.3, 197.4, 36.3, 153.9, 135.1, 103.6, 120.2, 151.9, 141.6, 131.7, 322.5, 356.7, 81.4, 168.9, 334.9, 245.3, 321.5, 125.9, 272.8, 135.3, 80.7, 122.4, 328.6, 95.9, 309.2, 247.8, 311.1, 324.0, 281.9, 316.3, 184.4, 247.5, 270.9, 302.7, 280.1, 4.6, 275.0, 219.0, 220.3, 353.8, 220.5, 197.6, 176.6, 152.6, 91.8, 92.0, 113.5, 359.4, 205.3, 159.1, 247.1, 30.7, 357.7, 333.3, 351.8, 329.8, 50.5, 257.9, 153.0, 102.4, 236.9, 328.6, 311.5, 284.7, 349.7, 165.3, 197.4, 36.3, 153.9, 135.1, 103.6, 120.2, 151.9, 141.6, 131.7, 322.5, 356.7, 81.4]
R1_pref_changes = [16.2, 148.4, 80.0, 82.7, -52.8, 6.6, -39.8, -127.0, -45.3, 117.9, -55.0, 35.4, 226.4, 259.6, 292.2, 257.8, 51.6, -15.1, 174.2, -9.5, 220.3, 227.5, -74.6, 243.9, 178.9, 24.6, 216.5, 3.6, -43.3, -39.7, -64.6, -25.7, 12.1, -203.7, 295.2, 27.8, 34.9, 5.5, -4.3, 338.4, 306.3, 15.2, -19.3, -206.9, -18.5, -171.2, -144.6, 166.2, 75.1, 32.6, 146.1, 10.2, 3.7, -92.8, -303.4, -174.9, -206.0, -211.7, -162.2, 144.0, 123.3, -225.1, 162.5, 263.3, 30.5, -16.2, -148.4, -80.0, -82.7, 52.8, -6.6, 39.8, 127.0, 45.3, -117.9, 55.0, -35.4, -226.4, -259.6, -292.2, -257.8, -51.6, 15.1, -174.2, 9.5, -220.3, -227.5, 74.6, -243.9, -178.9, -24.6, -216.5, -3.6, 43.3, 39.7, 64.6, 25.7, -12.1, 203.7, -295.2, -27.8, -34.9, -5.5, 4.3, -338.4, -306.3, -15.2, 19.3, 206.9, 18.5, 171.2, 144.6, -166.2, -75.1, -32.6, -146.1, -10.2, -3.7, 92.8, 303.4, 174.9, 206.0, 211.7, 162.2, -144.0, -123.3, 225.1, -162.5, -263.3, -30.5]

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