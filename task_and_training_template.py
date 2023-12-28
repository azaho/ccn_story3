import argparse, json, time, random, datetime
import hashlib, torch, math, pathlib, shutil, sys
import numpy as np
from torch import nn

verbose = True  # print info in console?
net_size = 100  # size of input layer and recurrent layer
hyperparameters = {
    "random_string": "AA",  # human-readable string used for random initialization (for reproducibility)
    "optimizer": "Adam",  # options: Adam
    "noise_amplitude": 0.1,  # normal noise with s.d. = noise_amplitude

    "clip_gradients": True,  # limit gradient size (allows the network to train for a long time without diverging)
    "max_gradient_norm": 10,

    # regularization options:
    #   L1_weights, L2_weights;
    #   L1_activity;
    #   SE_weights (spatial embedding / wiring cost, embeds units in a square grid, see Achterberg & Akarca et al. (2022));
    #   None.
    "regularization": "None",
    "regularization_lambda": 0,

    "batch_size": 64,
    "note_error_every_steps": 50,  # only relevant if verbose is True

    "train_for_steps": 5000,
    "save_network_every_steps": 5000,
    "learning_rate": 1e-3,
}
hyperparameters["random_seed"] = int(hashlib.sha1(hyperparameters["random_string"].encode("utf-8")).hexdigest(), 16) % 10**8  # random initialization seed (for reproducibility)
if hyperparameters["regularization"] is None or hyperparameters["regularization"].lower() == "none":
    hyperparameters["regularization_lambda"] = 0
    hyperparameters["regularization"] = "None"

task_parameters = {
    "task_name": "1DIR1O",
    "input_direction_units": net_size,  # how many direction-selective input units?

    "delay0_from": 10, "delay0_to": 20,  # range (inclusive) for lengths of variable delays (in timesteps)
    "delay1_from": 10, "delay1_to": 90,
    "delay2_from": 120, "delay2_to": 160,

    "show_direction_for": 10,  # in timesteps
    "show_cue_for": 100,  # in timesteps
    "dim_input": net_size + 1,  # plus one input for go cue signal
    "dim_output": 2,
    "distractor_probability": 1.0  # probability that the distractor will be present on any given trial
}

model_parameters = {
    "model_name": "sgtCTRNN",
    "dim_input": task_parameters["dim_input"],
    "dim_output": task_parameters["dim_output"],
    "dim_recurrent": net_size,
    "tau": 10,  # defines ratio tau/dt (see continuous-time recurrent neural networks)
    "nonlinearity": "retanh",  # options: retanh, tanh
    "input_bias": True,
    "output_bias": False,

    "connectivity_cos_exponent": 1,  # for hand-designed models
}

additional_comments = [
    "Training criterion: MSE loss",
    "Noise added at every timestep of the trial",
    "Inputs NOT discretized",
    "Output sin/cos",
]

pretrain_delays = {
    "delay0_from": 10, "delay0_to": 20,  # range (inclusive) for lengths of variable delays (in timesteps)
    "delay1_from": 10, "delay1_to": 30,
    "delay2_from": 60, "delay2_to": 80,
}
final_delays = {
    "delay0_from": 10, "delay0_to": 20,  # range (inclusive) for lengths of variable delays (in timesteps)
    "delay1_from": 10, "delay1_to": 90,
    "delay2_from": 120, "delay2_to": 160,
}

#%FINAL_PARAMETERS_HERE%

def update_directory_name():
    # directory for results to be saved to
    directory = "data/"
    directory += f"{model_parameters['model_name']}_{task_parameters['task_name']}"
    directory += f"_dr{model_parameters['dim_recurrent']}_n{hyperparameters['noise_amplitude']}"
    directory += f"_la{hyperparameters['regularization_lambda']}"
    if "shuffle_amount" in model_parameters: directory += f"_sa{model_parameters['shuffle_amount']}"
    if "legi_exponent" in model_parameters: directory += f"_le{model_parameters['legi_exponent']}"
    if "scale_factor" in model_parameters: directory += f"_sf{model_parameters['scale_factor']}"
    if "connectivity_cos_exponent" in model_parameters: directory += f"_e{model_parameters['connectivity_cos_exponent']}"
    if "stable_unit_proportion" in model_parameters: durectory += f"_sup{model_parameters['stable_unit_proportion']}"
    directory += f"_dp{task_parameters['distractor_probability']}"
    directory += f"_r{hyperparameters['random_string']}"
    directory += "/"  # needs to end with a slash
    return directory
directory = update_directory_name()

def update_random_seed():
    hyperparameters["random_seed"] = int(hashlib.sha1(hyperparameters["random_string"].encode("utf-8")).hexdigest(),
                                         16) % 10 ** 8  # random initialization seed (for reproducibility)
    if hyperparameters["regularization"] is None or hyperparameters["regularization"].lower() == "none" or \
            hyperparameters["regularization_lambda"] == 0:
        hyperparameters["regularization_lambda"] = 0
        hyperparameters["regularization"] = "None"
    random.seed(hyperparameters["random_seed"])
    torch.manual_seed(hyperparameters["random_seed"])
    np.random.seed(hyperparameters["random_seed"])
update_random_seed()

def save_analysis_notebooks(directory, args):
    # copy this script, analysis ipynb, and util script into the same directory
    # for easy importing in the jupyter notebook
    shutil.copy("task_and_training_template.py", directory + "task_and_training_template.py")
    shutil.copy(sys.argv[0], directory + "task_and_training.py")
    # copy the analysis notebook
    shutil.copy("analysis_and_figures.ipynb", directory + "analysis_and_figures.ipynb")
    # replace parsed args with their values in the copied file (for analysis)
    with open(directory + "task_and_training.py", "r+") as f:
        data = f.read()
        parser_start = data.index("# PARSER START")
        parser_end = data.index("# PARSER END")
        data = data[0:parser_start:] + data[parser_end::]
        for arg in vars(args):
            replace = f"args.{arg}"
            replaceWith = f"{getattr(args, arg)}"
            if type(getattr(args, arg)) == str:
                replaceWith = '"' + replaceWith + '"'
            data = data.replace(replace, replaceWith)
        data = data.replace("#" + "%FINAL_PARAMETERS_HERE%",
f"""
task_parameters["delay1_from"] = {task_parameters["delay1_from"]}
task_parameters["delay1_to"] = {task_parameters["delay1_to"]}
task_parameters["delay2_from"] = {task_parameters["delay2_from"]}
task_parameters["delay2_to"] = {task_parameters["delay2_to"]}
task_parameters["distractor_probability"] = {task_parameters["distractor_probability"]}
""")
        f.seek(0)
        f.write(data)
        f.truncate()

def save_training_data(directory, result):
    error_store, error_store_o1, error_store_o2, gradient_norm_store = result["error_store"], result["error_store_o1"], \
                                                                       result["error_store_o2"], result[
                                                                           "gradient_norm_store"]
    training_dynamics = torch.cat((
        error_store.unsqueeze(0),
        error_store_o1.unsqueeze(0),
        error_store_o2.unsqueeze(0),
        gradient_norm_store.unsqueeze(0)
    ))
    torch.save(training_dynamics, directory + "training_dynamics.pt")

def save_metadata(directory, task, model, result, path=None, verbose=True):
    if path is None: path = directory + "info.json"
    _path = pathlib.Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)
    # save all parameters
    info = {
        "hyperparameters": hyperparameters,
        "task_parameters": task_parameters,
        "model_parameters": model_parameters,
        "additional_comments": additional_comments,
        "directory": directory,

        "errors_last": result["errors_last"],
        "errors_best": result["errors_best"],
        "error_distractor": task.evaluate_model(model, distractor_probability=1, noise_amplitude=hyperparameters["noise_amplitude"], direction_resolution=30),  # final error of network
        "error_nodistractor": task.evaluate_model(model, distractor_probability=0, noise_amplitude=hyperparameters["noise_amplitude"], direction_resolution=30),  # final error of network
        "error_store_saved": result["error_store_saved"]  # MSE of the network on every training step where the weights were saved
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
    if verbose:
        print(info["error_distractor"])


# (only for hand-designed networks)
# local-excitation, global-inhibition function to use for the pattern
# pref1 and pref2 are preferred directions between the units, in degrees
# for exponent=1, this is regular cosine connectivity; For larger values of the exponent,
# this becomes a narrower tuning curve.
def _legi(pref1, pref2, exponent):
    return ((0.5+0.5*torch.cos((pref1-pref2)/180 * torch.pi))**exponent)
def _legi_integral(exponent):
    dx = 1
    x = torch.arange(-180, 180, dx)
    y = _legi(x, 0, exponent=exponent)
    return torch.sum(y) * dx / 360
def legi(pref1, pref2, exponent=None):
    if exponent is None: exponent = model_parameters["connectivity_cos_exponent"]
    return _legi(pref1, pref2, exponent=exponent) - _legi_integral(exponent)


class Task:
    # outputs mask defining which timesteps noise should be applied to
    # for a given choice of (delay0, delay1, delay2)
    # output is (total_time, )
    def get_noise_mask(self, delay0, delay1, delay2):
        noise_from_t = delay0 + task_parameters["show_direction_for"] * 2 + delay1
        noise_to_t = noise_from_t + delay2
        total_t = noise_to_t + task_parameters["show_cue_for"]
        mask = torch.zeros(total_t)
        mask[noise_from_t:noise_to_t] = 1
        mask[:] = 1
        return mask

    def get_median_delays(self):
        delay0 = (task_parameters["delay0_from"]+task_parameters["delay0_to"])//2
        delay1 = (task_parameters["delay1_from"]+task_parameters["delay1_to"])//2
        delay2 = (task_parameters["delay2_from"]+task_parameters["delay2_to"])//2
        return delay0, delay1, delay2

    # direction tuning curve for input cells. Based on:
    # Andrew Teich & Ning Qian (2003) "Learning and adaptation in a recurrent model of V1 direction selectivity"
    def _o_spikes(self, pref, stim, exponent, max_spike, k):
        # o_spikes: spike numbers per trial for direction tuning cells
        # r = o_spikes(pref, stim, exponent, k)
        # pref: row vec for cells' preferred directions
        # stim: column vec for stimulus directions
        # exponent: scalar determining the widths of tuning. larger value for sharper tuning
        # maxSpike: scalar for mean max spike number when pref = stim
        # k: scalar for determining variance = k * mean
        # spikes: different columuns for cells with different pref orintations
        #         different rows for different stim directions
        np_ = pref.shape[0]  # number of elements in pref
        ns = stim.shape[0]  # number of elements in stim
        prefs = torch.ones((ns, 1)) @ pref[None, :]  # ns x np array, (ns x 1) @ (1 x np)
        stims = stim[:, None] @ torch.ones((1, np_))  # ns x np array, (ns x 1) @ (1 x np)
        # mean spike numbers
        mean_spike = max_spike * (0.5 * (torch.cos( (prefs - stims)) + 1)) ** exponent  # ns x np array
        # sigma for noise
        sigma_spike = torch.sqrt(k * mean_spike)
        # spikes = normrnd(meanSpike, sigmaSpike)# ns x np array, matlab
        spikes = torch.normal(mean_spike, sigma_spike)  # ns x np array, python
        # no negative spike numbers
        spikes[spikes < 0] = 0  # ns x np array
        return spikes

    # convert input direction angle (in deg) to firing rates of direction-selective input units
    def _input_direction_representation(self, direction):
        pref = 2*math.pi * torch.arange(task_parameters["input_direction_units"]) / task_parameters["input_direction_units"]
        stim = torch.tensor([(direction / 180 * math.pi)], dtype=torch.float32)
        exponent = 4; max_spike = 1; k = 0
        rates = self._o_spikes(pref, stim, exponent, max_spike, k)[0]
        return rates

    # convert target output direction angles (in deg) to target firing rates of output units
    def _output_direction_representation(self, direction1, direction2):
        rates = torch.zeros(2)
        theta = direction1 / 180 * math.pi
        rates[0] = math.sin(theta)
        rates[1] = math.cos(theta)
        return rates

    # generate parameters for a trial
    # (make choices for directions and delay lengths)
    # can pass parameters to leave them unchanged
    def choose_trial_parameters(self, direction1=None, direction2=None, delay0=None, delay1=None, delay2=None):
        if direction1 is None: direction1 = random.random() * 360
        if direction2 is None: direction2 = random.random() * 360
        if delay0 is None: delay0 = random.randint(task_parameters["delay0_from"], task_parameters["delay0_to"])
        if delay1 is None: delay1 = random.randint(task_parameters["delay1_from"], task_parameters["delay1_to"])
        if delay2 is None: delay2 = random.randint(task_parameters["delay2_from"], task_parameters["delay2_to"])
        return direction1, direction2, delay0, delay1, delay2

    # generate one trial of the task (there will be batch_size of them in the batch)
    # direction1 and direction2 in degrees
    # output tensors: input, target, mask (which timesteps to include in the loss function)
    def _make_trial(self, direction1, direction2, delay0, delay1, delay2, distractor_probability=0):
        # generate the tensor of inputs
        i_direction1 = torch.zeros(task_parameters["dim_input"])
        i_direction1[:task_parameters["input_direction_units"]] = self._input_direction_representation(direction1)
        i_direction1 = i_direction1.repeat(task_parameters["show_direction_for"], 1)
        i_direction2 = torch.zeros(task_parameters["dim_input"])
        i_direction2[:task_parameters["input_direction_units"]] = self._input_direction_representation(direction2)
        i_direction2 = i_direction2.repeat(task_parameters["show_direction_for"], 1)
        i_direction2 = i_direction2*(1 if random.random() < distractor_probability else 0)
        i_delay0 = torch.zeros((delay0, task_parameters["dim_input"]))
        i_delay2 = torch.zeros((delay2, task_parameters["dim_input"]))

        t1 = delay1
        t2 = t1 + task_parameters["show_direction_for"]
        i_delay2[t1:t2] = i_direction2

        i_cue = torch.zeros((task_parameters["show_cue_for"], task_parameters["dim_input"]))
        i_cue[:, -1] = 1
        i_full = torch.cat((i_delay0, i_direction1, i_delay2, i_cue))  # (total_time, dim_input)

        o_beforecue = torch.zeros(task_parameters["show_direction_for"] + delay0 + delay2, task_parameters["dim_output"])
        o_cue = self._output_direction_representation(direction1, direction2).repeat(task_parameters["show_cue_for"], 1)
        o_full = torch.cat((o_beforecue, o_cue))  # (total_time, dim_output)

        b_mask = torch.cat((torch.zeros((task_parameters["show_direction_for"] + delay0 + delay2,)),
                             torch.ones((task_parameters["show_cue_for"],))))  # (total_time,)

        return i_full, o_full, b_mask

    # generate a batch (of size batch_size)
    # all trials in batch have the same (delay0, delay1, delay2) but direction1 and direction2 vary (are random)
    # returns shapes (batch_size, total_time, dim_input), (batch_size, total_time, dim_output), (batch_size, total_time)
    def make_random_directions_batch(self, batch_size, delay0, delay1, delay2, distractor_probability=0):
        batch = []  # inputs in the batch
        batch_labels = []  # target outputs in the batch
        output_masks = []  # masks in the batch
        for j in range(batch_size):
            direction1, direction2, *_ = self.choose_trial_parameters(None, None, delay0, delay1, delay2)
            i_full, o_full, b_mask = self._make_trial(direction1, direction2, delay0, delay1, delay2, distractor_probability=distractor_probability)
            batch.append(i_full.unsqueeze(0))
            batch_labels.append(o_full.unsqueeze(0))
            output_masks.append(b_mask.unsqueeze(0))
        return torch.cat(batch), torch.cat(batch_labels), torch.cat(output_masks)

    # generate a batch (of size 180/resolution * 180/resolution)
    # all trials in batch have the same (delay0, delay1, delay2) but direction1 and direction2 vary (all int values, up to resolution)
    # returns shapes (batch_size, total_time, dim_input), (batch_size, total_time, dim_output), (batch_size, total_time)
    def make_all_integer_directions_batch(self, delay0, delay1, delay2, resolution=16, distractor_probability=0):
        batch = []  # inputs in the batch
        batch_labels = []  # target outputs in the batch
        output_masks = []  # masks in the batch
        for direction1 in np.arange(resolution)/resolution*360:
            for direction2 in np.arange(resolution)/resolution*360:
                i_full, o_full, b_mask = self._make_trial(direction1, direction2, delay0, delay1, delay2, distractor_probability=distractor_probability)
                batch.append(i_full.unsqueeze(0))
                batch_labels.append(o_full.unsqueeze(0))
                output_masks.append(b_mask.unsqueeze(0))
        return torch.cat(batch), torch.cat(batch_labels), torch.cat(output_masks)

    # convert sin, cos outputs to the angles they represent (normalizing outputs to have sum of squares = 1)
    # converts separately for every trial and timestep
    # output o1 and o2 are (batch_size, t_to-t_from)
    def convert_sincos_to_angles(self, output, t_from, t_to):
        trig = output[:, t_from:t_to, :]
        o1 = torch.atan2((trig[:, :, 0] / (trig[:, :, 0] ** 2 + trig[:, :, 1] ** 2) ** 0.5),
                         (trig[:, :, 1] / (trig[:, :, 0] ** 2 + trig[:, :, 1] ** 2) ** 0.5)) * 180 / math.pi
        if task_parameters["dim_output"] == 2:  # Do not remember second orientation
            o2 = o1*0
        else:
            o2 = torch.atan2((trig[:, :, 2] / (trig[:, :, 2] ** 2 + trig[:, :, 3] ** 2) ** 0.5),
                             (trig[:, :, 3] / (trig[:, :, 2] ** 2 + trig[:, :, 3] ** 2) ** 0.5)) * 180 / math.pi
        return o1, o2

    # calculate MSE error between output and target
    # calculates raw MSE and also sqrt(MSE) in degrees (after normalizing and converting to angles)
    def calculate_errors(self, target, output, mask, t_from, t_to):
        error = torch.mean((output[mask == 1] - target[mask == 1]) ** 2, dim=0)
        mse_o1 = (error[0] + error[1]).item() / 2
        if task_parameters["dim_output"] == 2: mse_o2 = mse_o1 * 0
        else: mse_o2 = (error[2] + error[3]).item() / 2
        o1_o, o2_o = self.convert_sincos_to_angles(output, t_from, t_to)
        o1_t, o2_t = self.convert_sincos_to_angles(target, t_from, t_to)
        error_o1 = torch.minimum(torch.minimum((o1_o - o1_t) ** 2, (o1_o - o1_t + 360) ** 2), (o1_o - o1_t - 360) ** 2)
        angle_error_o1 = torch.mean(error_o1).item() ** 0.5
        error_o2 = torch.minimum(torch.minimum((o2_o - o2_t) ** 2, (o2_o - o2_t + 360) ** 2), (o2_o - o2_t - 360) ** 2)
        angle_error_o2 = torch.mean(error_o2).item() ** 0.5
        return mse_o1, mse_o2, angle_error_o1, angle_error_o2

    # evaluate MSE and angle errors based on median delays, from the all integer direction batch
    def evaluate_model(self, model, noise_amplitude=0, direction_resolution=6, distractor_probability=0):
        # run the model on all possible directions
        ao_input, ao_target, ao_mask = self.make_all_integer_directions_batch(*self.get_median_delays(), direction_resolution, distractor_probability=distractor_probability)
        ao_noise_mask = self.get_noise_mask(*self.get_median_delays())
        ao_noise_mask = ao_noise_mask.repeat(ao_input.shape[0], 1).unsqueeze(2).repeat(1, 1, model.dim_recurrent)  # convert to (batch_size, total_time, dim_recurrent)
        ao_noise = torch.randn_like(ao_noise_mask) * ao_noise_mask * noise_amplitude
        ao_output, ao_h = model.forward(ao_input, noise=ao_noise)
        delay0, delay1, delay2 = self.get_median_delays()
        t5 = delay0 + delay2 + task_parameters["show_direction_for"]
        t6 = t5 + task_parameters["show_cue_for"]
        return self.calculate_errors(ao_target, ao_output, ao_mask, t5, t6)


class Task_outputO1O2(Task):
    # convert target output direction angles (in deg) to target firing rates of output units
    def _output_direction_representation(self, direction1, direction2):
        rates = torch.zeros(4)
        theta = direction1 / 180 * math.pi
        rates[0] = math.sin(theta)
        rates[1] = math.cos(theta)
        theta = direction2 / 180 * math.pi
        rates[2] = math.sin(theta)
        rates[3] = math.cos(theta)
        return rates


# continuous-time recurrent neural network (CTRNN)
# Tau * d(ah)/dt = -ah + W_h_ah @ f(ah) + W_ah_x @ x + b_ah
# Equation 1 from Miller & Fumarola 2012 "Mathematical Equivalence of Two Common Forms of Firing Rate Models of Neural Networks"
#
# ah[t] = ah[t-1] + (dt/Tau) * (-ah[t-1] + W_h_ah @ h[t−1] + W_x_ah @ x[t] + b_ah)
# h[t] = f(ah[t]) + noise[t], if noise_mask[t] = 1
# y[t] = W_h_y @ h[t] + b_y
#
# parameters to be learned: W_h_ah, W_x_ah, W_y_h, b_ah, b_y
# constants that are not learned: dt, Tau, noise
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dim_input = model_parameters["dim_input"]
        dim_output = model_parameters["dim_output"]
        dim_recurrent = model_parameters["dim_recurrent"]
        self.dim_input, self.dim_output, self.dim_recurrent = dim_input, dim_output, dim_recurrent
        self.dt, self.tau = 1, model_parameters["tau"]
        if model_parameters["nonlinearity"] == "tanh": self.f = torch.tanh
        if model_parameters["nonlinearity"] == "retanh": self.f = lambda x: torch.maximum(torch.tanh(x), torch.tensor(0))

        self.fc_x2ah = nn.Linear(dim_input, dim_recurrent, bias=model_parameters["input_bias"])  # W_ah_x @ x + b_ah
        self.fc_h2ah = nn.Linear(dim_recurrent, dim_recurrent, bias=False)  # W_h_ah @ h
        self.fc_h2y = nn.Linear(dim_recurrent, dim_output, bias=model_parameters["output_bias"])  # y = W_h_y @ h + b_y
        ah0 = torch.zeros(dim_recurrent)
        b_ah = torch.zeros(dim_recurrent)
        b_y = torch.zeros(dim_output)

        W_h_ah = torch.randn(dim_recurrent, dim_recurrent) / np.sqrt(dim_recurrent)
        W_x_ah = torch.randn(dim_recurrent, dim_input) / np.sqrt(dim_input)
        W_h_y = torch.zeros(dim_output, dim_recurrent)

        if model_parameters["input_bias"]: self.fc_x2ah.bias = torch.nn.Parameter(torch.squeeze(b_ah))
        if model_parameters["output_bias"]: self.fc_h2y.bias = torch.nn.Parameter(torch.squeeze(b_y))
        self.fc_x2ah.weight = torch.nn.Parameter(W_x_ah)  # W_x_ah @ x + b_ah
        self.fc_h2ah.weight = torch.nn.Parameter(W_h_ah)  # W_h_ah @ h
        self.fc_h2y.weight = torch.nn.Parameter(W_h_y)  # y = W_h_y @ h + b_y
        self.ah0 = torch.nn.Parameter(ah0, requires_grad=False)

    # output y and recurrent unit activations for all trial timesteps
    # input has shape (batch_size, total_time, dim_input) or (total_time, dim_input)
    # noise has shape (batch_size, total_time, dim_recurrent) or (total_time, dim_recurrent)
    def forward(self, input, noise):
        if len(input.shape) == 2:
            # if input has size (total_time, dim_input) (if there is only a single trial), add a singleton dimension
            input = input[None, :, :]  # (batch_size, total_time, dim_input)
            noise = noise[None, :, :]  # (batch_size, total_time, dim_recurrent)
        batch_size, total_time, dim_input = input.shape
        ah = self.ah0.repeat(batch_size, 1)
        h = self.f(ah)
        hstore = []  # store all recurrent activations at all timesteps. Shape (batch_size, total_time, dim_recurrent)
        for t in range(total_time):
            #print(self.fc_h2ah.weight.shape, self.fc_x2ah.weight.shape, input[:, t].shape)
            ah = ah + (self.dt / self.tau) * (-ah + self.fc_h2ah(h) + self.fc_x2ah(input[:, t]))
            h = self.f(ah) + noise[:, t, :]
            hstore.append(h)
        hstore = torch.stack(hstore, dim=1)
        output = self.fc_h2y(hstore)
        return output, hstore

    def save_firing_rates(self, task, path, resolution=8, noise_amplitude=None, distractor_probability=1):
        if noise_amplitude is None: noise_amplitude = 0

        delay0, delay1, delay2 = task.get_median_delays()
        delay1 = task_parameters["delay1_to"]  # maximum delay before distractor, to ensure convergence

        ao_input, ao_target, ao_mask = task.make_all_integer_directions_batch(delay0, delay1, delay2,
                                                                              distractor_probability=distractor_probability, resolution=resolution)
        ao_noise_mask = task.get_noise_mask(delay0, delay1, delay2)
        ao_noise_mask = ao_noise_mask.repeat(ao_input.shape[0], 1).unsqueeze(2).repeat(1, 1,
                                                                                       self.dim_recurrent)  # convert to (batch_size, total_time, dim_recurrent)
        ao_noise = torch.randn_like(ao_noise_mask) * ao_noise_mask * noise_amplitude  # no noise
        ao_output, ao_h = self.forward(ao_input, noise=ao_noise)
        arr = ao_h.reshape(-1, model_parameters["dim_recurrent"]).detach().numpy()

        _path = pathlib.Path(path)
        _path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)


# train the network on the task.
# outputs: error_store, error_store_o1, error_store_o2, gradient_norm_store
# error_store[j] -- the error after j parameter updates
# error_store_o1, error_store_o1 are errors in o1 and o2, respectively
# gradient_norm_store[j] -- the norm of the gradient after j parameter updates
def train_network(model, task, directory):
    def save_network(model, path, save_fr=True):
        _path = pathlib.Path(path)
        _path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state_dict': model.state_dict()}, path)
        if save_fr:
            model.save_firing_rates(task, path[:-4]+"_fr.npy")
    optimizer = None
    if hyperparameters["optimizer"].upper() == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    batch_size = hyperparameters["batch_size"]
    max_steps = hyperparameters["train_for_steps"]
    error_store = torch.zeros(max_steps + 1)
    error_store_o1 = torch.zeros(max_steps + 1)
    error_store_o2 = torch.zeros(max_steps + 1)
    error_store_saved = {}
    # error_store[0] is the error before any parameter updates have been made,
    # error_store[j] is the error after j parameter updates
    # error_store_o1, error_store_o1 are errors in o1 and o2, respectively
    gradient_norm_store = torch.zeros(max_steps + 1)
    # gradient_norm_store[0] is norm of the gradient before any parameter updates have been made,
    # gradient_norm_store[j] is the norm of the gradient after j parameter updates
    noise_amplitude = hyperparameters["noise_amplitude"]
    clip_gradients = hyperparameters["clip_gradients"]
    max_gradient_norm = hyperparameters["max_gradient_norm"]
    set_note_error = list(range(0, max_steps, hyperparameters["note_error_every_steps"]))
    if max_steps not in set_note_error: set_note_error.append(max_steps)
    set_note_error = np.array(set_note_error)
    set_save_network = list(range(0, max_steps, hyperparameters["save_network_every_steps"]))
    if max_steps not in set_save_network: set_save_network.append(max_steps)
    set_save_network = np.array(set_save_network)

    best_network_dict = None
    best_network_error = None
    for p in range(max_steps + 1):
        _, _, delay0, delay1, delay2 = task.choose_trial_parameters()  # choose the delays for this batch
        input, target, output_mask = task.make_random_directions_batch(batch_size, delay0, delay1, delay2, distractor_probability=task_parameters["distractor_probability"])
        noise_mask = task.get_noise_mask(delay0, delay1, delay2)
        noise_mask = noise_mask.repeat(batch_size, 1).unsqueeze(2).repeat(1, 1, model.dim_recurrent)  # convert to (batch_size, total_time, dim_recurrent)
        noise = torch.randn_like(noise_mask) * noise_mask * noise_amplitude
        output, h = model.forward(input, noise=noise)
        # output_mask: (batch_size, total_time, dim_output) tensor, elements
        # 0 (if timestep does not contribute to this term in the error function),
        # 1 (if timestep contributes to this term in the error function)

        error = torch.sum((output[output_mask == 1] - target[output_mask == 1]) ** 2, dim=0) / torch.sum(output_mask == 1)
        error_o1 = (error[0] + error[1]).item()
        if model_parameters["dim_output"] == 2:  # Do not remember the second orientation
            error_o2 = 0
        else: error_o2 = (error[2] + error[3]).item()
        error = torch.sum(error)

        # add regularization
        if hyperparameters["regularization"] == "L1_weights":
            for param in model.parameters():
                if param.requires_grad is True:
                    error += hyperparameters["regularization_lambda"] * torch.sum(torch.abs(param))
        elif hyperparameters["regularization"] == "L2_weights":
            for param in model.parameters():
                if param.requires_grad is True:
                    error += hyperparameters["regularization_lambda"] * torch.sum(param ** 2)
        elif hyperparameters["regularization"] == "SE_weights":  # embeds units in a square grid
            W = torch.abs(model.fc_h2ah.weight)

            net_size = model_parameters["dim_recurrent"]
            sqrt_net_size = int(net_size**0.5)
            d = torch.arange(sqrt_net_size).repeat(sqrt_net_size, 1)
            d = (d**2 + d.T**2)**0.5
            d = d.reshape(-1)

            s = torch.sum(W, dim=1) ** (-0.5)
            s = torch.diag(s)
            s = torch.matrix_exp(s @ W @ s)
            s.fill_diagonal_(0)

            error += hyperparameters["regularization_lambda"] * torch.sum(W * d * s)
        elif hyperparameters["regularization"] == "L1_activity":
            error += hyperparameters["regularization_lambda"] * torch.sum(torch.abs(h)) / np.prod(h.shape)

        error_store[p] = error.item()
        error_store_o1[p] = error_o1
        error_store_o2[p] = error_o2

        # don't train on step 0, just store error and initialize "best" network
        if p == 0:
            error_store_saved[0] = error.item()
            evaluation_thisnetwork = task.evaluate_model(model, distractor_probability=task_parameters["distractor_probability"], noise_amplitude=hyperparameters["noise_amplitude"])

            best_network_dict = model.state_dict()
            best_network_error = error.item()
            save_network(model, directory + f'model_best.pth', save_fr=False)
            evaluation_bestnetwork = evaluation_thisnetwork

            last_time = time.time()  # for estimating how long training will take
            continue
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the Tensors it will update (which are the learnable weights of the model)
        optimizer.zero_grad()
        # Backward pass: compute gradient of the error with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        error.backward()
        # clip the norm of the gradient
        if clip_gradients:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()

        # store gradient norms
        gradient = []  # store all gradients
        for param in model.parameters():  # model.parameters include those defined in __init__ even if they are not used in forward pass
            if param.requires_grad is True:  # model.parameters include those defined in __init__ even if param.requires_grad is False (in this case param.grad is None)
                gradient.append(param.grad.detach().flatten().cpu().numpy())
        gradient = np.concatenate(gradient)
        gradient_norm_store[p] = np.sqrt(np.sum(gradient ** 2)).item()
        # note running error in console
        if verbose and np.isin(p, set_note_error):
            error_wo_reg = torch.sum((output[output_mask == 1] - target[output_mask == 1]) ** 2) / torch.sum(
                output_mask == 1)
            print(f'{p} parameter updates: error = {error.item():.4f}, w/o reg {error_wo_reg.item():.4f}; o1 {error_o1:.4f}, o2 {error_o2:.4f}')
            passed_time = time.time() - last_time
            made_steps = hyperparameters["note_error_every_steps"]
            left_steps = max_steps - p
            left_time = left_steps / made_steps * passed_time
            print(f" = took {passed_time:.1f}s for {made_steps} steps, estimated time left {str(datetime.timedelta(seconds=int(left_time)))}")
            last_time = time.time()
            evaluation_thisnetwork = task.evaluate_model(model, distractor_probability=task_parameters["distractor_probability"], noise_amplitude=hyperparameters["noise_amplitude"])
            print(" = performance: ", *(f"{x:.4f}; " for x in evaluation_thisnetwork))
            print(" = best so far: ", *(f"{x:.4f}; " for x in evaluation_bestnetwork))

        # save network
        if np.isin(p, set_save_network):
            print("SAVING", f'model_parameterupdate{p}.pth')
            save_network(model, directory + f'model_parameterupdate{p}.pth')
            error_store_saved[p] = error.item()
        if evaluation_thisnetwork[0] < evaluation_bestnetwork[0]:
            best_network_dict = model.state_dict()
            best_network_error = error.item()
            save_network(model, directory + f'model_best.pth', save_fr=False)
            evaluation_bestnetwork = task.evaluate_model(model, distractor_probability=task_parameters["distractor_probability"], noise_amplitude=hyperparameters["noise_amplitude"])

    result = {
        "error_store": error_store,  # MSE error every training step
        "error_store_o1": error_store_o1,  # MSE error every training step (only O1=target component)
        "error_store_o2": error_store_o2,  # MSE error every training step (only O2=distractor component) -- only relevant if memorizing distractor too
        "gradient_norm_store": gradient_norm_store,    # backprop gradient norm error every training step
        "errors_last": evaluation_thisnetwork,
        "errors_best": evaluation_bestnetwork,
        "error_store_saved": error_store_saved    # MSE for every training step on which the weights are saved
    }
    return result


def output_connectivity_factors(directory, task, model, dir_prefix="data_json/"):
    """
        Performs the analysis determining the structural/functional ratios,
        and outputs them into a json and pdf file for this network.
    """
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

    def plot_recurrentweights(timestep, ax, lim=0.1, color="green", fit_curves=True):
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
        if fit_curves:
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

    def plot_inputweights(timestep, ax, color="green", lim=0.1, fit_curves=True):
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
        if fit_curves:
            ax.plot(x_binned, y_fit, "--", color=color, linewidth=3)
        return params

    fig = plt.figure(figsize=(8, 8))
    lim = 0.1
    plt.rc('font', **{'family': 'Arial', 'weight': 'normal', 'size': 17})
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_ylabel("weight")
    params_recurrent_cue = plot_recurrentweights((t1 + t2) // 2, ax1, color="k", lim=lim)
    params_recurrent_delay = plot_recurrentweights((t2 + t3) // 2, fig.add_subplot(2, 2, 2), color="k", lim=lim)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_ylabel("weight")
    params_input_cue = plot_inputweights((t1 + t2) // 2, ax3, color=targetinput_color, lim=lim)
    params_input_delay = plot_inputweights((t2 + t3) // 2, fig.add_subplot(2, 2, 4), color=distractor_color, lim=lim)
    plt.tight_layout()
    plt.savefig(dir_prefix+directory[5:-1] + "_connectivity.pdf", bbox_inches="tight")

    fig = plt.figure(figsize=(8, 8))
    plt.rc('font', **{'family': 'Arial', 'weight': 'normal', 'size': 17})
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_ylabel("weight")
    params_recurrent_cue = plot_recurrentweights((t1 + t2) // 2, ax1, color="k", lim=lim, fit_curves=False)
    params_recurrent_delay = plot_recurrentweights((t2 + t3) // 2, fig.add_subplot(2, 2, 2), color="k", lim=lim, fit_curves=False)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_ylabel("weight")
    params_input_cue = plot_inputweights((t1 + t2) // 2, ax3, color=targetinput_color, lim=lim, fit_curves=False)
    params_input_delay = plot_inputweights((t2 + t3) // 2, fig.add_subplot(2, 2, 4), color=distractor_color, lim=lim, fit_curves=False)
    plt.tight_layout()
    plt.savefig(dir_prefix+directory[5:-1] + "_connectivity_nofit.pdf", bbox_inches="tight")

    factors = {
        "structural_factor": params_recurrent_delay[0] / params_input_cue[0],
        "functional_factor": params_input_cue[0] / params_input_delay[0]
    }
    with open(dir_prefix+directory[5:-1] + '_connectivity.json', 'w') as f:
        json.dump(factors, f, indent=4)