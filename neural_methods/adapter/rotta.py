import torch
import torch.nn as nn
from utils import memory
from base_adapter import BaseAdapter
from copy import deepcopy
from base_adapter import softmax_entropy
from utils.bn_layers import RobustBN1d, RobustBN2d
from utils.utils import set_named_submodule, get_named_submodule
from utils.custom_transforms import get_tta_transforms
import sinc_aug, sinc_loss
import tent


class RoTTA(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(RoTTA, self).__init__(cfg, model, optimizer)
        self.mem = memory.CSTU(capacity=self.cfg.ADAPTER.RoTTA.MEMORY_SIZE, num_class=8, lambda_t=cfg.ADAPTER.RoTTA.LAMBDA_T, lambda_u=cfg.ADAPTER.RoTTA.LAMBDA_U)
        self.model_ema = self.build_ema(self.model)
        #self.transform = get_tta_transforms(cfg)

        self.transform = sinc_aug.apply_transformations
        self.nu = cfg.ADAPTER.RoTTA.NU
        self.update_frequency = cfg.ADAPTER.RoTTA.UPDATE_FREQUENCY  # actually the same as the size of memory bank
        self.current_instance = 0

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        # batch data
        with torch.no_grad():
            model.eval()
            self.model_ema.eval()
            ema_out = self.model_ema(batch_data)
            ema_smooth = tent.torch_detrend(torch.cumsum(ema_out, axis=0), torch.tensor(100.0))

            ema_smooth = ema_smooth.view(-1,180)

            fps = float(30)
            low_hz = float(0.66666667)
            high_hz = float(3.0)

            freqs, psd = tent.torch_power_spectral_density(ema_smooth, fps=fps, low_hz=low_hz, high_hz=high_hz,
                                                      normalize=False, bandpass=False)

            speed= 1.0

            pseudo_label = torch.round(freqs[psd.argmax(axis=1)]*3)

            entropy = []
            for i in range(pseudo_label.shape[0]):
                bandwidth_loss = sinc_loss.IPR_SSL(freqs, psd[i].unsqueeze(0), speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
                variance_loss = sinc_loss.EMD_SSL(freqs, psd[i].unsqueeze(0), speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
                sparsity_loss = sinc_loss.SNR_SSL(freqs, psd[i].unsqueeze(0), speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')

                entropy_ = bandwidth_loss + variance_loss + sparsity_loss
                entropy.append(entropy_)

            entropy = torch.stack(entropy)

        # add into memory

        batch_data = batch_data[:-1].view(-1,180,3,72,72)

        for i, data in enumerate(batch_data):
            p_l = int(pseudo_label[i].item())
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
            self.current_instance += 1

            if self.current_instance % self.update_frequency == 0:
                self.update_model(model, optimizer)

        return ema_out

    def update_model(self, model, optimizer):
        model.train()
        self.model_ema.train()
        # get memory data
        sup_data, ages = self.mem.get_memory()
        l_sup = None
        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data)
            num_of_gpu = 1
            class Auginfo:
                def __init__(self, frame_length):
                    self.aug_speed = 1
                    self.aug_flip = 1
                    self.aug_reverse = 1
                    self.aug_illum = 1
                    self.aug_gauss = 1
                    self.aug_resizedcrop = 1
                    self.frames_per_clip = frame_length  # 721
                    self.channels = 'rgb'
                    self.speed_slow = 0.6
                    self.speed_fast = 1.0  # self.speed_fast = 1.4

            auginfo = Auginfo(sup_data.shape[1]+1)

            instance_weight = timeliness_reweighting(ages)
            l_sup_list =[]
            for idx, sup_data_ in enumerate(sup_data):
                last_frame = torch.unsqueeze(sup_data_[-1, :, :, :], 0).repeat(num_of_gpu, 1, 1, 1)
                sup_data_ = torch.cat((sup_data_, last_frame), 0) # [T,C,H,W]
                strong_sup_aug , speed = self.transform(auginfo, sup_data_.permute(0, 2, 3, 1).cpu().numpy()) # [T,H,W,C]
                ema_sup_out = self.model_ema(sup_data_)
                stu_sup_out = model(strong_sup_aug.permute(1, 0, 2, 3))

                l_sup = sum(abs(stu_sup_out - ema_sup_out) * instance_weight[idx])
                l_sup_list.append(l_sup)

        l = l_sup.mean()
        print(l)
        if l is not None:
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        self.update_ema_variables(self.model_ema, self.model, self.nu)

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model

    def configure_model(self, model: nn.Module):

        model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer,
                                self.cfg.ADAPTER.RoTTA.ALPHA)
            momentum_bn.requires_grad_(True)
            set_named_submodule(model, name, momentum_bn)
        return model


def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))

