from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import torch
import torch.fft as fft

import neural_methods.augmentation.transforms as transforms
import neural_methods.augmentation.sinc_aug as sinc_aug
import neural_methods.loss.sinc_loss as sinc_loss

import matplotlib.pyplot as plt

from evaluation.post_process import _detrend, calculate_metric_per_video
from scipy.signal import butter
import scipy
import numpy as np
from neural_methods.utils.utils import set_named_submodule, get_named_submodule
import time

import wandb

WEIGHT_UPDATE = True

EPSILON = 1e-10
BP_LOW=2/3
BP_HIGH=3.0
BP_DELTA=0.1

import gc
def find_nearest_index(tensor_list, value):
    # 주어진 값과 리스트의 요소와의 차이를 계산하여 절대값을 취한 리스트 생성
    absolute_diff = [abs(x - value) for x in tensor_list]
    # 절대값이 가장 작은 인덱스 반환
    nearest_index = absolute_diff.index(min(absolute_diff))
    return nearest_index


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
    def __init__(self, model, optimizer, steps=1, episodic=False, model_name='None', b_scale=1.0 , s_scale=1.0,
                 v_scale=1.0, sm_scale=0.1 , fc_scale=5.0):
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

        self.log_list = []
        self.model_name = model_name
        self.tta_fw_cnt = 0
        self.tta_bw_cnt = 0
        self.iter =0
        self.freq_error_list = []
        self.total_loss_list =[]

        self.b_scale = b_scale
        self.s_scale = s_scale
        self.v_scale = v_scale
        self.sm_scale = sm_scale
        self.fc_scale = fc_scale

    def forward(self, x , y):
        if self.episodic:
            self.reset()

        loss_list = []
        error_list = []
        index_list = []

        for index in range(self.steps):
            outputs ,(fw_cnt,bw_cnt), (freq_error,total_loss) = forward_and_adapt\
                (x, y , self.model, self.optimizer, self.model_name, self.iter,self.b_scale,self.s_scale,self.v_scale,
                 self.sm_scale,self.fc_scale)

            self.tta_fw_cnt += fw_cnt
            self.tta_bw_cnt += bw_cnt
            self.iter+=1

            """
            if(self.iter %100 == 0):
                self.reset()
            """
            self.freq_error_list.append(freq_error)
            self.total_loss_list.append(total_loss)

            loss_list.append(total_loss)
            error_list.append(freq_error)
            index_list.append(index)

        index_list = np.array(index_list)
        color_map = plt.get_cmap('tab10')
        colors = color_map(index_list)

        """
        plt.scatter(x=np.array(loss_list), y=np.array(error_list),color=colors)
        plt.show()
        """


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


def get_filtered_freqs_psd(y):
    fps= 30
    low_hz = float(0.66666667)
    high_hz = float(3.0)

    y_ = torch_detrend(torch.cumsum(y, axis=0), torch.tensor(100.0))

    # bandpass filter between [0.75, 2.5] Hz
    # equals [45, 150] beats per min
    [b, a] = butter(1, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
    y_ = scipy.signal.filtfilt(b, a, np.double(y_.detach().cpu().numpy()))

    y_tensor = torch.from_numpy(y_.copy())
    freqs, y_psd = torch_power_spectral_density(y_tensor, fps=fps, low_hz=low_hz, high_hz=high_hz,
                                                normalize=True, bandpass=False)

    return freqs,y_psd

def get_loss(x,model):

    fps = float(30)
    low_hz = float(0.66666667)
    high_hz = float(3.0)

    predictions_ = model(x)
    predictions_smooth = torch_detrend(torch.cumsum(predictions_, axis=0), torch.tensor(100.0))

    # predictions_batch = predictions_smooth.view(-1,180)
    predictions_batch = predictions_smooth.T

    freqs, psd = torch_power_spectral_density(predictions_batch, fps=fps, low_hz=low_hz, high_hz=high_hz,
                                              normalize=False, bandpass=False)
    speed = torch.tensor([1.0])

    bandwidth_loss = sinc_loss.IPR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
    variance_loss = sinc_loss.EMD_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
    sparsity_loss = sinc_loss.SNR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')

    batch_total_loss = bandwidth_loss + sparsity_loss +variance_loss
    test_loss = batch_total_loss.detach().cpu().numpy()

    return test_loss


def get_loss_3d(x,model):

    fps = float(30)
    low_hz = float(0.66666667)
    high_hz = float(3.0)

    predictions_, _, _, _  = model(x)
    predictions_smooth = torch_detrend(torch.cumsum(predictions_.T, axis=0), torch.tensor(100.0))

    # predictions_batch = predictions_smooth.view(-1,180)
    predictions_batch = predictions_smooth.T

    freqs, psd = torch_power_spectral_density(predictions_batch, fps=fps, low_hz=low_hz, high_hz=high_hz,
                                              normalize=False, bandpass=False)
    speed = torch.tensor([1.0] * psd.shape[0])

    bandwidth_loss = sinc_loss.IPR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
    variance_loss = sinc_loss.EMD_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
    sparsity_loss = sinc_loss.SNR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')

    batch_total_loss = bandwidth_loss + sparsity_loss # +variance_loss
    test_loss = batch_total_loss.detach().cpu().numpy()

    return test_loss


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, y, model, optimizer, model_name ,iter, b_scale, s_scale,v_scale,sm_scale,fc_scale):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """

    fw_cnt = 0
    bw_cnt = 0

    total_loss = 0.0

    MC_DROP = 0
    SINC = 0
    SINC_ORIGIN = 0
    SINC_3d = 0

    AUG = 1
    ADAPTIVE_OPT = 0


    if model_name == 'EfficientPhys':
        SINC_ORIGIN = 1
    elif model_name == 'Physnet':
        SINC_3d = 1


    fps = float(30)
    low_hz = float(0.66666667)
    high_hz = float(3.0)
    # Save the initial parameters
    #initial_params = [param.clone().detach() for param in model.parameters()]

    # Save the initial parameters along with their names
    #initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}

    predictions_ = model(x)


    if SINC_3d :
        iter_num = 1


    """
    model.eval()
    loss_ = get_loss(x, model)
    model.train()
    """

    """
    for name, param in model.named_parameters():
        if has_parameter_changed(initial_params[name], param):
            print(f"Parameter '{name}' has changed during training.")
        else:
            print(f"Parameter '{name}' remains unchanged.")
    """


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
            self.aug_speed = 0
            self.aug_flip = 1
            self.aug_reverse = 0
            self.aug_illum = 0
            self.aug_gauss = 0
            self.aug_resizedcrop = 1
            self.frames_per_clip = frame_length #721
            self.channels = 'rgb'
            self.speed_slow = 0.6
            self.speed_fast = 1.0 # self.speed_fast = 1.4

    if SINC_ORIGIN:
        # model.train()
        auginfo = Auginfo(x.shape[0])

        x_aug, speed = sinc_aug.apply_transformations(auginfo, x.permute(0, 2, 3, 1).cpu().numpy())  # [C,T,H,W]
        predictions = model(x_aug.permute(1, 0, 2, 3))
        predictions_smooth = torch_detrend(torch.cumsum(predictions, axis=0), torch.tensor(100.0))


        # predictions_batch = predictions_smooth.view(-1, 180)
        predictions_batch = predictions_smooth.T

        freqs, psd = torch_power_spectral_density(predictions_batch, fps=fps, low_hz=low_hz, high_hz=high_hz,
                                                  normalize=False, bandpass=False)

        #speed = torch.tensor([speed]*4)
        speed = torch.tensor([speed])

        bandwidth_loss = sinc_loss.IPR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
        variance_loss = sinc_loss.EMD_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
        sparsity_loss = sinc_loss.SNR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
        print("bandwidth_loss:", bandwidth_loss, "sparsity_loss:", sparsity_loss, "variance_loss:", variance_loss)

        sinc_total_loss = bandwidth_loss + variance_loss + sparsity_loss
        total_loss += sinc_total_loss


    if SINC:
        #model.train()

        prediction_list = []

        iter_num = 1
        aug_num = 1
        window_size = 180
        auginfo = Auginfo(window_size+1)

        x_copy = x[:-1].view(-1, window_size, 3, 72, 72).clone()
        bs = x_copy.shape[0]

        speed_list = []


        if aug_num >0:
            for idx in range(bs):
                for iter in range(aug_num):
                    data_test = x_copy[idx]
                    last_frame = torch.unsqueeze(data_test[-1, :, :, :], 0).repeat(1, 1, 1, 1)
                    data_test = torch.cat((data_test, last_frame), 0)
                    x_aug, speed = sinc_aug.apply_transformations(auginfo, data_test.permute(0,2,3,1).cpu().numpy()) # [T,H,W,C]
                    predictions = model(x_aug.permute(1,0,2,3))
                    predictions_smooth = torch_detrend(torch.cumsum(predictions, axis=0), torch.tensor(100.0))
                    prediction_list.append(predictions_smooth)

                    speed_list.append(speed)
        else:
            for iter in range(iter_num):
                predictions_smooth = torch_detrend(torch.cumsum(predictions_, axis=0), torch.tensor(100.0))
                prediction_list.append(predictions_smooth)

                for _ in range(bs):
                    speed_list.append(1.0)


        prediction_list = torch.stack(prediction_list)

        predictions_batch = prediction_list.view(-1, window_size)
        #predictions_batch = predictions_smooth.T

        freqs, psd = torch_power_spectral_density(predictions_batch, fps=fps, low_hz=low_hz, high_hz=high_hz,
                                                  normalize=False, bandpass=False)
        speed = torch.tensor(speed_list)

        bandwidth_loss = sinc_loss.IPR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
        variance_loss = sinc_loss.EMD_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
        sparsity_loss = sinc_loss.SNR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
        print ("bandwidth_loss:",bandwidth_loss,"sparsity_loss:",sparsity_loss,"variance_loss:",variance_loss)

        sinc_total_loss  = bandwidth_loss + variance_loss + sparsity_loss
        total_loss += sinc_total_loss

    if SINC_3d:

        iter_count = 0
        iter_max = 1

        while(1):

            if iter_count >= iter_max:
                break

            total_loss = 0.0
            auginfo = Auginfo(x.shape[2])

            prediction_list = []
            speed_list = []

            for data in x:
                if AUG:
                    x_aug, speed = sinc_aug.apply_transformations(auginfo, data.cpu().numpy())  # [C,T,H,W]
                    speed_list.append(speed)
                    predictions, _, _, _  = model(x_aug.unsqueeze(0).cuda())
                else:
                    speed_list.append(1.0)
                    predictions, _, _, _ = model(data.unsqueeze(0))

                predictions = (predictions - torch.mean(predictions)) / torch.std(predictions)  # normalize

                predictions_smooth = torch_detrend(torch.cumsum(predictions.T, axis=0), torch.tensor(100.0))
                prediction_list.append(predictions_smooth)
            fw_cnt += 1

            prediction_list = torch.stack(prediction_list)
            predictions_batch = prediction_list.view(-1, 128)
            speed = torch.tensor(speed_list)

            freqs, psd = torch_power_spectral_density(predictions_batch, fps=fps, low_hz=low_hz, high_hz=high_hz,
                                                      normalize=True, bandpass=False)

            bandwidth_loss = sinc_loss.IPR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
            variance_loss = sinc_loss.EMD_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
            sparsity_loss = sinc_loss.SNR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')

            sinc_total_loss = bandwidth_loss *b_scale + sparsity_loss *s_scale + variance_loss *v_scale

            frequency_const_loss = torch.var(psd,dim=0).sum() * fc_scale
            if WEIGHT_UPDATE:
                total_loss += sinc_total_loss
                total_loss += frequency_const_loss
            else:
                sinc_total_loss = sinc_total_loss.detach()


            #if ADAPTIVE_OPT and iter_count==0 and total_loss > 0.6:
            #    iter_max = 10


            if ADAPTIVE_OPT and iter_count==0:
                if total_loss >= 0.9:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 1e-2
                else:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 1e-3


            y_ = torch_detrend(torch.cumsum(y, axis=0), torch.tensor(100.0))

            # bandpass filter between [0.75, 2.5] Hz
            # equals [45, 150] beats per min
            [b, a] = butter(1, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
            y_ = scipy.signal.filtfilt(b, a, np.double(y_.detach().cpu().numpy()))

            y_tensor = torch.from_numpy(y_.copy())
            freqs, y_psd = torch_power_spectral_density(y_tensor, fps=fps, low_hz=low_hz, high_hz=high_hz,
                                                     normalize=True, bandpass=False)

            freq_error = (torch.mean(abs(freqs[y_psd.argmax(dim=1)] - freqs[psd.argmax(dim=1)]))).detach().cpu().numpy()

            iter_count+=1

            p_target = scipy.signal.filtfilt(b, a, np.double(predictions_batch.detach().cpu().numpy()))

            if WEIGHT_UPDATE:
                temporal_smooth_loss = (torch.mean(abs(torch.from_numpy(p_target.copy()).cuda()-predictions_batch))
                                        * sm_scale)
                total_loss += temporal_smooth_loss

            wandb.log( {'sinc_total_loss': sinc_total_loss, 'bandwidth_loss': bandwidth_loss,
                        "sparsity_loss:": sparsity_loss, "variance_loss:": variance_loss,
                        "freq_error": freq_error, "frequency_const_loss": frequency_const_loss,
                        "temporal_smooth_loss":temporal_smooth_loss}, step=iter)

            print("bandwidth_loss:", round(bandwidth_loss.item(),2), "sparsity_loss:", round(sparsity_loss.item(),2),
                  "variance_loss:", round(variance_loss.item(),2) ,"frequency_const_loss:", round(frequency_const_loss.item(),2),
                  "temporal_smooth_loss:", round(temporal_smooth_loss.item(),2))

            if WEIGHT_UPDATE:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()


            if iter_count>=1:
                """
                fig = plt.figure()
                error = abs(freqs[y_psd[0].argmax(dim=0)] - freqs[psd[0].argmax(dim=0)]).detach().cpu().numpy()
                title = f"iter_count:{iter_count},error:{round(float(error),2)}," \
                        f"b_l:{round(float(bandwidth_loss.detach().cpu().numpy()),2)}," \
                        f"s_l:{round(float(sparsity_loss.detach().cpu().numpy()),2)}," \
                        f"v_l:{round(float(variance_loss.detach().cpu().numpy()),2)}"
                plt.title(title)
                plt.plot(psd[0].detach().cpu().numpy(),color='blue')
                plt.plot(y_psd[0].detach().cpu().numpy(),color='red')
                plt.axvline(x=psd[0].detach().cpu().numpy().argmax(), color='blue', linestyle='--')
                plt.axvline(x=y_psd[0].detach().cpu().numpy().argmax(), color='red', linestyle='--')

                plt.axvline(x=120, color='gray', linestyle='-')
                plt.axvline(x=540, color='gray', linestyle='-')

                plt.show()
                plt.close()
                plt.clf()
                
                plt.plot(psd[0].detach().cpu().numpy())
                plt.plot(psd[1].detach().cpu().numpy())
                plt.plot(psd[2].detach().cpu().numpy())
                plt.plot(psd[3].detach().cpu().numpy())
                plt.show()
            
                print (torch.var(psd))
                
                plt.savefig("output_psd.png")
                plt.close()
                plt.clf()
                gc.collect()
                wandb.log({
                    "output_psd": [
                        wandb.Image('output_psd.png')
                    ]
                },step=iter)
                
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        wandb.log({f'weights/{name}': wandb.Histogram(param.clone().detach().cpu().numpy())},step=iter)
                """
            bw_cnt += 1





    if SINC_ORIGIN or SINC :
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
    """
    model.eval()
    predictions_new = model(x)
    loss_new = get_loss(x, model)

    if loss_ >  loss_new:
        last_prediction = predictions_new
        print ('new predict', loss_-loss_new)
    else:
        last_prediction = predictions_
    """

    return predictions_,(fw_cnt,bw_cnt), (freq_error, total_loss.detach().cpu().numpy())


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        #if isinstance(m, nn.BatchNorm2d):
        for np, p in m.named_parameters():
            if p.requires_grad:
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

    if WEIGHT_UPDATE:
        model.train()
    else:
        model.eval()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for name, m in model.named_modules():

        print (name)
        """
        if name.find('ConvBlock1') != -1:
            target_layer = get_named_submodule(model, name)
            target_layer.requires_grad_(True)
        """
        if isinstance(m, nn.BatchNorm2d) or isinstance(m,nn.BatchNorm3d):
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