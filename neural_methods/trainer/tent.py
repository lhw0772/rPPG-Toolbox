from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import torch
import torch.fft as fft

import transforms
import sinc_aug
import sinc_loss

import matplotlib.pyplot as plt

from evaluation.post_process import _detrend, calculate_metric_per_video
from scipy.signal import butter
import scipy
import numpy as np


EPSILON = 1e-10
BP_LOW=2/3
BP_HIGH=3.0
BP_DELTA=0.1


def torch_detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = torch.eye(signal_length).cuda()
    ones = torch.ones(signal_length)
    minus_twos = -2 * torch.ones(signal_length)
    diags_data = torch.stack([ones, minus_twos, ones])
    diags_index = torch.tensor([0, 1, 2])

    D = torch.sparse.spdiags(diags_data, diags_index, (signal_length, signal_length)).cuda()
    D = D.to_dense()
    D = D[:-2]

    detrended_signal = torch.matmul(
        (H - torch.inverse(H + (lambda_value ** 2) * torch.matmul(D.T, D))), input_signal)
    return detrended_signal


def normalize_psd(psd):
    return psd / torch.sum(psd, keepdim=True, dim=1) ## treat as probabilities


def ideal_bandpass(freqs, psd, low_hz, high_hz):
    freq_idcs = torch.logical_and(freqs >= low_hz, freqs <= high_hz)
    freqs = freqs[freq_idcs]
    psd = psd[:,freq_idcs]
    return freqs, psd



def torch_power_spectral_density(x, nfft=5400, fps=90, low_hz=BP_LOW, high_hz=BP_HIGH, return_angle=False, radians=True, normalize=True, bandpass=True):
    centered = x - torch.mean(x, keepdim=True, dim=1)
    rfft_out = fft.rfft(centered, n=nfft, dim=1)
    psd = torch.abs(rfft_out)**2
    N = psd.shape[1]
    freqs = fft.rfftfreq(2*N-1, 1/fps).cuda()
    if return_angle:
        angle = torch.angle(rfft_out)
        if not radians:
            angle = torch.rad2deg(angle)
        if bandpass:
            freqs, psd, angle = ideal_bandpass(freqs, psd, low_hz, high_hz, angle=angle)
        if normalize:
            psd = normalize_psd(psd)
        return freqs, psd, angle
    else:
        if bandpass:
            freqs, psd = ideal_bandpass(freqs, psd, low_hz, high_hz)
        if normalize:
            psd = normalize_psd(psd)
        return freqs, psd


def add_noise_to_constants(predictions):
    B,T = predictions.shape
    for b in range(B):
        if torch.allclose(predictions[b][0], predictions[b]): # constant volume
            predictions[b] = torch.rand(T) - 0.5
    return predictions


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x , y):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, y , self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def get_smooth_result(input):

    fps= 30
    predictions =_detrend(np.cumsum(input.T.detach().cpu()),100)
    [b, a] = butter(1, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
    predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
    predictions = torch.unsqueeze(torch.tensor(predictions.copy(),requires_grad=True), 0).cuda()
    return predictions


def mc_dropout_uncertainty_loss(outputs, num_mc_samples):
    variance = torch.var(outputs, dim=0, unbiased=False)  # Calculate variance across batches
    return torch.mean(variance)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, y, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """

    total_loss = 0.0

    MC_DROP = 0
    SINC = 1


    fps = float(30)
    low_hz = float(0.66666667)
    high_hz = float(3.0)
    # Save the initial parameters
    #initial_params = [param.clone().detach() for param in model.parameters()]

    # Save the initial parameters along with their names
    initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}


    predictions_ = model(x)


    if MC_DROP:
        num_mc_samples = 4

        outputs = []
        for _ in range(num_mc_samples):
            outputs_ = model(x)
            outputs_ = torch_detrend(torch.cumsum(outputs_, axis=0), torch.tensor(100.0))
            outputs.append(outputs_)

        outputs = torch.stack(outputs)
        mc_loss = mc_dropout_uncertainty_loss(outputs, num_mc_samples)
        print ("mc_loss:", mc_loss)

        total_loss+= mc_loss


    class Auginfo:
        def __init__(self, frame_length):
            self.aug_speed = 1
            self.aug_flip = 1
            self.aug_reverse = 1
            self.aug_illum = 1
            self.aug_gauss = 1
            self.aug_resizedcrop = 1
            self.frames_per_clip = frame_length #721
            self.channels = 'rgb'
            self.speed_slow = 0.6
            self.speed_fast = 1.0 # self.speed_fast = 1.4

    if SINC:
        auginfo = Auginfo(x.shape[0])

        x_aug, speed = sinc_aug.apply_transformations(auginfo, x.permute(0,2,3,1).cpu().numpy()) # [C,T,H,W]
        predictions = model(x_aug.permute(1,0,2,3))

        predictions_smooth = torch_detrend(torch.cumsum(predictions, axis=0), torch.tensor(100.0))

        #predictions_batch = predictions_smooth.view(-1,180)
        predictions_batch = predictions_smooth.T

        freqs, psd = torch_power_spectral_density(predictions_batch, fps=fps, low_hz=low_hz, high_hz=high_hz,
                                                  normalize=False, bandpass=False)

        bandwidth_loss = sinc_loss.IPR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
        variance_loss = sinc_loss.EMD_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
        sparsity_loss = sinc_loss.SNR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
        print ("bandwidth_loss:",bandwidth_loss,"sparsity_loss:",sparsity_loss,"variance_loss:",variance_loss)

        sinc_total_loss  = bandwidth_loss + variance_loss + sparsity_loss

        total_loss += sinc_total_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


    """
    for name, param in model.named_parameters():
        if has_parameter_changed(initial_params[name], param):
            print(f"Parameter '{name}' has changed during training.")
        else:
            print(f"Parameter '{name}' remains unchanged.")
    """
    return predictions_


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"


def has_parameter_changed(initial, current):
    return not torch.all(torch.eq(initial, current))