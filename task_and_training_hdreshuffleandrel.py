import argparse, json, time, random, datetime
import hashlib, torch, math, pathlib, shutil, sys
import numpy as np
from torch import nn
from task_and_training_template import *

# PARSER START
parser = argparse.ArgumentParser(description='Train networks')
parser.add_argument('--net_size', type=int, help='size of input layer and recurrent layer', default=100)
parser.add_argument('--random', type=str, help='human-readable string used for random initialization', default="AA")
parser.add_argument('--shuffle_amount', type=float, help='how much to shift tunings?', default=0)
parser.add_argument('--scale_factor', type=float, help='determines ratio between input-ring and ring-ring connection strengths', default=1)
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
    "model_name": "hdreshufflerelCTRNN",
    "dim_recurrent": args.net_size,
    "dim_input": args.net_size + 1,  # plus one input for go cue signal
    "shuffle_amount": args.shuffle_amount,
    "scale_factor": args.scale_factor,
})
additional_comments += [
    "Reshuffle of tuning network, training is on top-level parameters + output layer"
]

#%FINAL_PARAMETERS_HERE%

directory = update_directory_name()
update_random_seed()

R1_i = torch.arange(model_parameters["dim_recurrent"])
R1_pref = R1_i/model_parameters["dim_recurrent"]*360
R1_pref_changes = [random.randint(-model_parameters["shuffle_amount"], model_parameters["shuffle_amount"]) for _ in R1_i]


# Modification of the class Model -- constrain the architecture to this particular solution class
class Model(Model):
    def __init__(self):
        super().__init__()

        del self.fc_x2ah, self.fc_h2ah
        self.b_ah = torch.zeros(self.dim_recurrent)
        self.b_y = torch.zeros(self.dim_output)

        self.R1_pref_changes = torch.tensor(R1_pref_changes)
        self.R1_i = R1_i
        self.R1_pref = R1_pref
        self.IN_pref = torch.arange(task_parameters["input_direction_units"])/task_parameters["input_direction_units"]*360

        # TRAINABLE PARAMETERS:
        # 1: R1->R1 and input->R1 curve magnitudes
        # 2: R1 bias
        self.top_parameters = nn.Parameter(torch.tensor([2/args.scale_factor**0.5, -1])/args.net_size*10)

    # output y and recurrent unit activations for all trial timesteps
    # input has shape (batch_size, total_time, dim_input) or (total_time, dim_input)
    # noise has shape (batch_size, total_time, dim_recurrent) or (total_time, dim_recurrent)
    def forward(self, input, noise):
        # build matrices based on top-level parameters
        self.W_h_ah = legi(self.R1_pref.repeat(len(self.R1_pref), 1), self.R1_pref.repeat(len(self.R1_pref), 1).T) * self.top_parameters[0] * model_parameters["scale_factor"]
        self.W_x_ah = legi((self.R1_pref+self.R1_pref_changes).repeat(task_parameters["input_direction_units"], 1).T, self.IN_pref.repeat(len(R1_pref), 1)) * self.top_parameters[0]
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


def output_connectivity_factors(dir_prefix="data_json/"):
    # writes the structural/functional connectivity factors and the corresponding figures to files

    import json, torch, math
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    import scipy.stats
    from matplotlib.ticker import FormatStrFormatter
    from scipy.optimize import curve_fit
    analyze_network = "final"  # options: "best", "final", <parameter update step no>
    noise_amplitude = 0.1  # if run analyses with noise, noise amplitude
    show_figures = True  # True if running in jupyter notebook; False if running a .py file
    running_from_data = True  # True if code is running in the data folder next to the model.pth. if false, must run from training file
    distractor_probability = 1.0

    ############################################ (denotes code block delimiter)

    resolution = 30
    ORI_RES = 360 // resolution
    ORI_SET = torch.arange(0, 360, ORI_RES)
    ORI_SET_SIZE = ORI_SET.shape[0]

    # fix delays at median values for analysis
    delay0, delay1, delay2 = task.get_median_delays()
    delay1 = task_parameters["delay1_to"]  # max delay1 (to ensure convergence to final state for analysis)
    show_direction_for = task_parameters["show_direction_for"]
    show_cue_for = task_parameters["show_cue_for"]
    total_time = show_direction_for + show_cue_for + delay0 + delay2
    t1, t1d = delay0, "before O1 presented"
    t1_5, t1_5d = delay0 + show_direction_for // 2, "amid 01 presentation"
    t2, t2d = delay0 + show_direction_for, "after O1 presented"
    t3, t3d = delay0 + show_direction_for + delay1, "before O2 presented"
    t3_5, t3_5d = delay0 + show_direction_for + delay1 + show_direction_for // 2, "amid O2 presentation"
    t4, t4d = delay0 + show_direction_for + delay1 + show_direction_for, "after O2 presented"
    t5, t5d = delay0 + show_direction_for + delay2, "before go cue"
    t6, t6d = total_time - 1, "at end of task"

    # run the model on all possible directions
    ao_input, ao_target, ao_mask = task.make_all_integer_directions_batch(delay0, delay1, delay2,
                                                                          distractor_probability=distractor_probability,
                                                                          resolution=resolution)
    ao_noise_mask = task.get_noise_mask(delay0, delay1, delay2)
    ao_noise_mask = ao_noise_mask.repeat(ao_input.shape[0], 1).unsqueeze(2).repeat(1, 1,
                                                                                   model.dim_recurrent)  # convert to (batch_size, total_time, dim_recurrent)
    ao_noise = torch.randn_like(ao_noise_mask) * ao_noise_mask * noise_amplitude
    ao_output, ao_h = model.forward(ao_input, noise=ao_noise)

    # output model errors (with noise and without)
    mse_o1, mse_o2, err_o1, err_o2 = task.calculate_errors(ao_target, ao_output, ao_mask, t5, t6)
    ao_output_nn, ao_h_nn = model.forward(ao_input, noise=ao_noise * 0)
    mse_o1_nn, mse_o2_nn, err_o1_nn, err_o2_nn = task.calculate_errors(ao_target, ao_output_nn, ao_mask, t5, t6)

    # for every timestep and every unit, calculate its activity in all trials
    ao_data = torch.zeros((total_time, model.dim_recurrent, ORI_SET_SIZE, ORI_SET_SIZE))
    for direction1 in range(ORI_SET_SIZE):
        for direction2 in range(ORI_SET_SIZE):
            o = ao_h[direction1 * ORI_SET_SIZE + direction2]
            ao_data[:, :, direction1, direction2] = o

    # detach from autograd
    ao_output = ao_output.detach()
    ao_h = ao_h.detach()
    ao_data = ao_data.detach()

    ############################################ (denotes code block delimiter)

    timestep, timestep_description = t5, t5d
    cutoff_criterion = "box"  # options: ratio, box
    ring_cutoff = 2  # if ratio: minumum variance ratio to consider unit a ring unit
    min_pri_var = 0.25  # if box: minimum variance in primary direction to consider unit a ring unit
    max_sec_var = 0.20  # if box: maximum variance in the other direction to consider unit a ring unit

    var_1 = torch.var(torch.mean(ao_data[timestep], dim=2), dim=1) ** 0.5 + 0.01
    var_2 = torch.var(torch.mean(ao_data[timestep], dim=1), dim=1) ** 0.5 + 0.01

    if cutoff_criterion == "ratio":
        R1_i = torch.where(var_1 / var_2 > ring_cutoff)[0]
        R2_i = torch.where(var_2 / var_1 > ring_cutoff)[0]
        DT_i = torch.tensor([x for x in range(model.dim_recurrent) if (x not in R1_i) and (x not in R2_i)], dtype=int)
    elif cutoff_criterion == "box":
        R1_i = torch.where(torch.logical_and(var_1 > min_pri_var, var_2 < max_sec_var))[0]
        R2_i = torch.where(torch.logical_and(var_2 > min_pri_var, var_1 < max_sec_var))[0]
        DT_i = torch.tensor([x for x in range(model.dim_recurrent) if (x not in R1_i) and (x not in R2_i)], dtype=int)

    ############################################ (denotes code block delimiter)

    def calc_pref(units_i, timestep=t5, to=1, data=None, round_prefs=False):
        if data is None: data = ao_data
        w = torch.sum(data[timestep][units_i], dim=3 - to).detach().numpy()
        a = np.angle(
            np.sum(w * np.exp(1j * (np.arange(ORI_SET_SIZE) / ORI_SET_SIZE * 2 * np.pi)).reshape(1, -1), axis=1) / (
                        np.sum(np.abs(w), axis=1) + 0.01)) * 180 / np.pi
        a[a < 0] = a[a < 0] + 360
        a = torch.tensor(a)
        if round_prefs: a = torch.round(a)
        return a

    prefs_1 = []  # every unit's preferred O1
    prefs_2 = []  # every unit's preferred O2
    strengths_1 = []  # every unit's preferred O1
    strengths_2 = []  # every unit's preferred O2
    for timestep in range(total_time):
        prefs_1.append(calc_pref(range(model.dim_recurrent), timestep=timestep, to=1).unsqueeze(0))
        prefs_2.append(calc_pref(range(model.dim_recurrent), timestep=timestep, to=2).unsqueeze(0))
    prefs_1 = torch.cat(prefs_1)
    prefs_2 = torch.cat(prefs_2)

    # sort units according to their preferred directions (don't sort DT)
    R1_pref = prefs_1[t5 - 1][R1_i]
    R2_pref = prefs_2[t5 - 1][R2_i]
    R1_i = R1_i.clone()[torch.argsort(R1_pref)]
    R1_pref = R1_pref.clone()[torch.argsort(R1_pref)]
    R2_i = R2_i.clone()[torch.argsort(R2_pref)]
    R2_pref = R2_pref.clone()[torch.argsort(R2_pref)]
    order_indices = torch.cat((R1_i, DT_i, R2_i))

    ############################################ (denotes code block delimiter)

    targetinput_color = "#006838"
    distractor_color = "#97211F"

    # function that will be fit to the averaged connection weights
    def cosine_fit(x, a, b):
        return a * np.cos(x * np.pi / 180) + b

    def get_connplot_graph(units1_id=None, unit1_pref=None, units2_id=None, unit2_pref=None, sm=0):
        weight_matrix = None  # different models may store weights differently
        try:
            weight_matrix = model.fc_h2ah.weight
        except:
            weight_matrix = model.W_h_ah
        distances_weights = {}
        distances = []
        weights = []
        for i in range(len(units1_id)):
            for j in range(len(units2_id)):
                for k in range(-sm // 2, sm // 2 + 1):
                    if j == i: continue
                    diff = (unit2_pref[j] - unit1_pref[i]).item()
                    if diff > 180: diff -= 360
                    if diff < -180: diff += 360
                    diff += k
                    w_ij = weight_matrix[units2_id[j], units1_id[i]]
                    distances.append(diff)
                    weights.append(w_ij.item())
        return np.array(distances), np.array(weights)

    def plot_recurrentweights(timestep, ax, lim=0.1, color="green"):
        x, y = get_connplot_graph(R1_i, calc_pref(R1_i, timestep, to=1), R1_i, calc_pref(R1_i, timestep, to=1))
        bins = np.linspace(-180, 180, 20)
        x_binned = []
        y_binned = []
        sem_binned = []
        bin_ids = np.digitize(x, bins)
        for i in range(1, len(bins)):
            x_binned.append(np.mean(x[bin_ids == i]))
            y_binned.append(np.mean(y[bin_ids == i]))
            sem_binned.append(scipy.stats.sem(y[bin_ids == i]))
        x_binned, y_binned, sem_binned = np.array(x_binned), np.array(y_binned), np.array(sem_binned)
        confint_binned = sem_binned * 1.96
        ax.axhline(y=0.0, color='k', linestyle='--', linewidth=2)
        ax.fill_between(x_binned, y_binned - confint_binned, y_binned + confint_binned, color=color, alpha=0.3,
                        linewidth=0)
        ax.plot(x_binned, y_binned, "-", color=color, linewidth=3)
        ax.set_xlim(-180, 180)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(-lim * 1.1, lim * 1.1)
        ax.set_yticks([-lim, 0, lim])
        ax.set_yticks([])
        ax.set_xticks([-180, 0, 180])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d°'))
        # ax.set_xlabel("∆ preferred angle")
        # ax.set_ylabel("weight")
        ax.set_yticks([-lim, 0, lim])

        params, covariance = curve_fit(cosine_fit, x_binned, y_binned)
        y_fit = cosine_fit(x_binned, *params)
        ax.plot(x_binned, y_fit, "--", color=color, linewidth=3)
        return params

    def get_connplot_iu_graph(units_id, unit_pref, sm=0):
        weight_matrix = None  # different models may store weights differently
        try:
            weight_matrix = model.fc_x2ah.weight
        except:
            weight_matrix = model.W_x_ah
        distances_weights = {}
        distances = []
        weights = []
        for i in range(len(units_id)):
            for j in range(task_parameters["input_direction_units"]):
                for k in range(-sm // 2, sm // 2 + 1):
                    # if j == i: continue
                    diff = (unit_pref[i] - round(360 * j / task_parameters["input_direction_units"])).item()
                    if diff > 180: diff -= 360
                    if diff < -180: diff += 360
                    diff += k

                    w_ij = weight_matrix[units_id[i], j]
                    distances.append(diff)
                    weights.append(w_ij.item())
        return np.array(distances), np.array(weights)

    def plot_inputweights(timestep, ax, color="green", lim=0.1):
        x, y = get_connplot_iu_graph(R1_i, calc_pref(R1_i, timestep, to=1))
        bins = np.linspace(-180, 180, 20)
        x_binned = []
        y_binned = []
        sem_binned = []
        bin_ids = np.digitize(x, bins)
        for i in range(1, len(bins)):
            x_binned.append(np.mean(x[bin_ids == i]))
            y_binned.append(np.mean(y[bin_ids == i]))
            sem_binned.append(scipy.stats.sem(y[bin_ids == i]))
        x_binned, y_binned, sem_binned = np.array(x_binned), np.array(y_binned), np.array(sem_binned) * 1.96
        # ax.scatter(x, y, 5, color="gray", alpha=0.3)
        ax.axhline(y=0.0, color='k', linestyle='--', linewidth=2)
        ax.fill_between(x_binned, y_binned - sem_binned, y_binned + sem_binned, color=color, alpha=0.3, linewidth=0)
        ax.plot(x_binned, y_binned, "-", color=color, linewidth=3)
        ax.set_xlim(-180, 180)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(-lim * 1.1, lim * 1.1)
        ax.set_yticks([-lim, 0, lim])
        ax.set_yticks([])
        ax.set_xticks([-180, 0, 180])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d°'))
        ax.set_xlabel("∆ preferred angle")
        # ax.set_ylabel("weight")
        ax.set_yticks([-lim, 0, lim])

        params, covariance = curve_fit(cosine_fit, x_binned, y_binned)
        y_fit = cosine_fit(x_binned, *params)
        ax.plot(x_binned, y_fit, "--", color=color, linewidth=3)
        return params

    fig = plt.figure(figsize=(8, 8))
    lim = 0.2
    plt.rc('font', **{'family': 'DejaVu Sans', 'weight': 'normal', 'size': 17})
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_ylabel("weight")
    params_recurrent_cue = plot_recurrentweights((t1 + t2) // 2, ax1, color="b", lim=lim)
    params_recurrent_delay = plot_recurrentweights((t2 + t3) // 2, fig.add_subplot(2, 2, 2), color="b", lim=lim)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_ylabel("weight")
    params_input_cue = plot_inputweights((t1 + t2) // 2, ax3, color=targetinput_color, lim=lim)
    params_input_delay = plot_inputweights((t2 + t3) // 2, fig.add_subplot(2, 2, 4), color=distractor_color, lim=lim)
    plt.tight_layout()

    plt.savefig(dir_prefix+directory[5:-1] + "_connectivity.pdf", bbox_inches="tight")
    factors = {
        "structural_factor": params_recurrent_delay[0] / params_input_cue[0],
        "functional_factor": params_input_cue[0] / params_input_delay[0]
    }
    with open(dir_prefix+directory[5:-1] + '_connectivity.json', 'w') as f:
        json.dump(factors, f, indent=4)


if __name__ == "__main__":
    # train the network and save weights
    task = Task()
    model = Model()

    hyperparameters["train_for_steps"] = 100

    directory = update_directory_name()
    result = train_network(model, task, directory)

    save_metadata(directory, task, model, result, path="data_json/"+directory[5:-1]+".json")
    output_connectivity_factors()

    # for now, nothing else is necessary
    # save_metadata(directory, task, model, result)
    # save_training_data(directory, result)
    # model.save_firing_rates(task, "data_npy/" + directory[5:-1] + ".npy")
    # save_metadata(directory, task, model, result, path="data_npy/" + directory[5:-1] + ".json")
    # save_analysis_notebooks(directory, args)