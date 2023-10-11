import torch
import math
import random
from torch import nn

net_size = 100  # size of the recurrent layer of the RNN (only used here to scale the input dimensionality)
task_parameters = {
    "task_name": "1DIR1O",
    "input_direction_units": net_size,  # how many direction-selective input units?

    "delay0_from": 20, "delay0_to": 40,  # range (inclusive) for lengths of variable delays (in timesteps)
    "delay1_from": 10, "delay1_to": 90,
    "delay2_from": 120, "delay2_to": 160,

    "show_direction_for": 10,  # in timesteps
    "show_cue_for": 100,  # in timesteps
    "dim_input": net_size + 1,  # plus one input for go cue signal
    "dim_output": 2,
    "distractor_visible": True  # train with or without distractor (distractor = O2)?
}

# not used in this file; just for information. Pretraining on shorter delays for quicker convergence
pretrain_delays = {
    "delay0_from": 20, "delay0_to": 40,  # range (inclusive) for lengths of variable delays (in timesteps)
    "delay1_from": 10, "delay1_to": 30,
    "delay2_from": 60, "delay2_to": 80,
}


class Task:
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
        mean_spike = max_spike * (0.5 * (torch.cos((prefs - stims)) + 1)) ** exponent  # ns x np array
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
    #   of shapes (total_time, dim_input); (total_time, dim_output); (total_time,)
    def _make_trial(self, direction1, direction2, delay0, delay1, delay2, show_distractor=True):
        # generate the tensor of inputs
        i_direction1 = torch.zeros(task_parameters["dim_input"])
        i_direction1[:task_parameters["input_direction_units"]] = self._input_direction_representation(direction1)
        i_direction1 = i_direction1.repeat(task_parameters["show_direction_for"], 1)
        i_direction2 = torch.zeros(task_parameters["dim_input"])
        i_direction2[:task_parameters["input_direction_units"]] = self._input_direction_representation(direction2)
        i_direction2 = i_direction2.repeat(task_parameters["show_direction_for"], 1)
        i_direction2 = i_direction2 * (1 if show_distractor else 0)
        i_delay0 = torch.zeros((delay0, task_parameters["dim_input"]))
        i_delay2 = torch.zeros((delay2, task_parameters["dim_input"]))

        t1 = delay1
        t2 = t1 + task_parameters["show_direction_for"]
        i_delay2[t1:t2] = i_direction2

        i_cue = torch.zeros((task_parameters["show_cue_for"], task_parameters["dim_input"]))
        i_cue[:, -1] = 1
        i_full = torch.cat((i_delay0, i_direction1, i_delay2, i_cue))  # (total_time, dim_input)

        o_beforecue = torch.zeros(task_parameters["show_direction_for"] + delay0 + delay2,
                                  task_parameters["dim_output"])
        o_cue = self._output_direction_representation(direction1, direction2).repeat(
            task_parameters["show_cue_for"], 1)
        o_full = torch.cat((o_beforecue, o_cue))  # (total_time, dim_output)

        b_mask = torch.cat((torch.zeros((task_parameters["show_direction_for"] + delay0 + delay2,)),
                            torch.ones((task_parameters["show_cue_for"],))))  # (total_time,)

        return i_full, o_full, b_mask


if __name__=="__main__":
    task = Task()
    direction1, direction2, delay0, delay1, delay2 = task.choose_trial_parameters()
    trial = task._make_trial(direction1, direction2, delay0, delay1, delay2, show_distractor = True)

    trial_input, trial_target, trial_mask = trial
    print(trial)