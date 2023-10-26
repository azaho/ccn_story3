import argparse, json, time, random, datetime
import hashlib, torch, math, pathlib, shutil, sys
import numpy as np
from torch import nn
from task_and_training_template import *

# PARSER START
parser = argparse.ArgumentParser(description='Train networks')
parser.add_argument('--net_size', type=int, help='size of input layer and recurrent layer', default=net_size)
parser.add_argument('--random', type=str, help='human-readable string used for random initialization', default="AA")
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
    "model_name": "hdgating_and_inversion_asymmetricweightsCTRNN",
    "dim_recurrent": args.net_size,
    "dim_input": args.net_size + 1,  # plus one input for go cue signal
})
additional_comments += [
    "Gating+Inversion network, training is on top-level parameters + output layer"
]

#%FINAL_PARAMETERS_HERE%

directory = update_directory_name()
update_random_seed()

R1a_si, R1a_ei = 0, model_parameters["dim_recurrent"]//4
R1a_i = torch.arange(R1a_si, R1a_ei)
R1b_si, R1b_ei = model_parameters["dim_recurrent"]//4, (model_parameters["dim_recurrent"]//4)*2
R1b_i = torch.arange(R1b_si, R1b_ei)
DT_si, DT_ei = (model_parameters["dim_recurrent"]//4)*2, model_parameters["dim_recurrent"]
DT_i = torch.arange(DT_si, DT_ei)
R1a_pref = R1a_i/len(R1a_i)*360
R1b_pref = torch.arange(len(R1b_i))/len(R1b_i)*360
DT_pref = DT_i/len(DT_i)*360


# Modification of the class Model -- constrain the architecture to this particular solution class
class Model(Model):
    def __init__(self):
        super().__init__()

        del self.fc_x2ah, self.fc_h2ah
        self.b_ah = torch.zeros(self.dim_recurrent)
        self.b_y = torch.zeros(self.dim_output)

        self.R1a_si, self.R1a_ei = R1a_si, R1a_ei
        self.R1a_i = R1a_i
        self.R1b_si, self.R1b_ei = R1b_si, R1b_ei
        self.R1b_i = R1b_i
        self.DT_si, self.DT_ei = DT_si, DT_ei
        self.DT_i = DT_i
        self.R1a_pref = R1a_pref
        self.R1b_pref = R1b_pref
        self.DT_pref = DT_pref
        self.IN_pref = torch.arange(task_parameters["input_direction_units"])/task_parameters["input_direction_units"]*360

        # TRAINABLE PARAMETERS:
        # 1: input-DT, DT-R1, R1-R1 curve magnitude
        # 2: R1 bias and DT bias
        # 3: R1-DT nonspecific connection magnitude
        # 4: how much R1a-R1b and R1b-R1a curve magnitude differ from intra
        self.top_parameters = nn.Parameter(torch.tensor([3, -0.5, -0.5, 1])/args.net_size*10)

    # output y and recurrent unit activations for all trial timesteps
    # input has shape (batch_size, total_time, dim_input) or (total_time, dim_input)
    # noise has shape (batch_size, total_time, dim_recurrent) or (total_time, dim_recurrent)
    def forward(self, input, noise):
        # build matrices based on top-level parameters
        W_h_ah = torch.zeros((self.dim_recurrent, self.dim_recurrent), dtype=torch.float32)
        W_x_ah = torch.zeros((self.dim_recurrent, self.dim_input), dtype=torch.float32)
        b_ah = torch.zeros_like(self.b_ah)
        W_h_ah[R1a_si:R1a_ei, R1a_si:R1a_ei] = legi(self.R1a_pref.repeat(len(self.R1a_pref), 1), self.R1a_pref.repeat(len(self.R1a_pref), 1).T) * self.top_parameters[0]
        W_h_ah[R1b_si:R1b_ei, R1b_si:R1b_ei] = legi(self.R1b_pref.repeat(len(self.R1b_pref), 1), self.R1b_pref.repeat(len(self.R1b_pref), 1).T) * self.top_parameters[0]
        W_h_ah[R1b_si:R1b_ei, R1a_si:R1a_ei] = legi(self.R1a_pref.repeat(len(self.R1b_pref), 1), self.R1b_pref.repeat(len(self.R1a_pref), 1).T) * (-self.top_parameters[0]-self.top_parameters[3])
        W_h_ah[R1a_si:R1a_ei, R1b_si:R1b_ei] = legi(self.R1b_pref.repeat(len(self.R1a_pref), 1), self.R1a_pref.repeat(len(self.R1b_pref), 1).T) * (-self.top_parameters[0]+self.top_parameters[3])

        W_h_ah[R1a_si:R1a_ei, DT_si:DT_ei] = legi(self.R1a_pref.repeat(len(self.DT_pref), 1).T, self.DT_pref.repeat(len(self.R1a_pref), 1)) * self.top_parameters[0]
        W_h_ah[R1b_si:R1b_ei, DT_si:DT_ei] = legi(self.R1b_pref.repeat(len(self.DT_pref), 1).T, self.DT_pref.repeat(len(self.R1b_pref), 1)) * self.top_parameters[0]

        W_h_ah[DT_si:DT_ei, R1a_si:R1a_ei] = torch.ones((len(DT_i), len(R1a_i))) * self.top_parameters[2]
        W_h_ah[DT_si:DT_ei, R1b_si:R1b_ei] = torch.ones((len(DT_i), len(R1b_i))) * self.top_parameters[2]

        W_x_ah[DT_si:DT_ei, :-1] = legi(self.DT_pref.repeat(task_parameters["input_direction_units"], 1).T, self.IN_pref.repeat(len(self.DT_pref), 1)) * self.top_parameters[0]
        b_ah[R1a_si:R1a_ei] = torch.ones((len(R1a_i),)) * self.top_parameters[1]
        b_ah[R1b_si:R1b_ei] = torch.ones((len(R1b_i),)) * self.top_parameters[1]
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