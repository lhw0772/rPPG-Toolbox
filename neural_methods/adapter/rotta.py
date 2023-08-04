import torch
import torch.nn as nn

import config
from neural_methods.utils import memory
from neural_methods.adapter.base_adapter import BaseAdapter
from copy import deepcopy
from neural_methods.adapter.base_adapter import softmax_entropy
from neural_methods.utils.bn_layers import RobustBN1d, RobustBN2d, RobustBN3d
from neural_methods.utils.utils import set_named_submodule, get_named_submodule
from neural_methods.utils.custom_transforms import get_tta_transforms
import neural_methods.augmentation.sinc_aug as sinc_aug
import neural_methods.loss.sinc_loss as sinc_loss
import neural_methods.trainer.tent as tent
import matplotlib.pyplot as plt


def batched_normed_psd(x, y):
  """
  x: M(# aug) x T(# time stamp)
  y: M(# aug) x T(# time stamp)
  """

  return torch.matmul(x[:,:,0],y[:,:,0].T).unsqueeze(0)

def label_distance(labels_1, labels_2 , label_temperature=0.1):
  # labels: bsz x M(#augs)
  # output: bsz x M(#augs) x M(#augs)

  labels_1 = labels_1.unsqueeze(0)
  labels_2 = labels_2.unsqueeze(0)

  dist_mat = - torch.abs(labels_1[:, :, None] - labels_2[:, None, :])
  prob_mat = torch.softmax(dist_mat / label_temperature,dim =-1)
  return prob_mat

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
        INPUT_3D = 1

        if INPUT_3D:
            with torch.no_grad():
                model.eval()
                self.model_ema.eval()
                ema_out, _, _, _   = self.model_ema(batch_data)
                ema_smooth = tent.torch_detrend(torch.cumsum(ema_out, axis=1), torch.tensor(100.0))

                #ema_smooth = ema_smooth.view(-1,180)

                fps = float(30)
                low_hz = float(0.66666667)
                high_hz = float(3.0)

                freqs, psd = tent.torch_power_spectral_density(ema_smooth, fps=fps, low_hz=low_hz, high_hz=high_hz,
                                                          normalize=False, bandpass=False)

                speed= torch.tensor([torch.tensor(1.0)])
                pseudo_label = torch.round(freqs[psd.argmax(axis=1)]*3)-2

                entropy = []
                for i in range(pseudo_label.shape[0]):
                    bandwidth_loss = sinc_loss.IPR_SSL(freqs, psd[i].unsqueeze(0), speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
                    variance_loss = sinc_loss.EMD_SSL(freqs, psd[i].unsqueeze(0), speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
                    sparsity_loss = sinc_loss.SNR_SSL(freqs, psd[i].unsqueeze(0), speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')

                    entropy_ = bandwidth_loss + variance_loss + sparsity_loss
                    entropy.append(entropy_)

                entropy = torch.stack(entropy)

            # add into memory
            # batch_data_ = batch_data[:-1].view(-1,180,3,72,72)

            for i, data in enumerate(batch_data):
                p_l = int(pseudo_label[i].item())

                if p_l >= 8 or p_l < 0:
                    continue

                uncertainty = entropy[i].item()
                current_instance = (data, p_l, uncertainty)
                self.mem.add_instance(current_instance)
                self.current_instance += 1

                if self.current_instance % self.update_frequency == 0:
                    for i in range(1):
                        self.update_model(model, optimizer)

            #ema_out = self.model_ema(batch_data)

            return ema_out

        else:
            with torch.no_grad():
                model.eval()
                self.model_ema.eval()
                ema_out = self.model_ema(batch_data)
                ema_smooth = tent.torch_detrend(torch.cumsum(ema_out, axis=0), torch.tensor(100.0))

                ema_smooth = ema_smooth.view(-1, 180)

                fps = float(30)
                low_hz = float(0.66666667)
                high_hz = float(3.0)

                freqs, psd = tent.torch_power_spectral_density(ema_smooth, fps=fps, low_hz=low_hz, high_hz=high_hz,
                                                               normalize=False, bandpass=False)

                speed = torch.tensor([torch.tensor(1.0)])
                pseudo_label = torch.round(freqs[psd.argmax(axis=1)] * 3) - 2

                entropy = []
                for i in range(pseudo_label.shape[0]):
                    bandwidth_loss = sinc_loss.IPR_SSL(freqs, psd[i].unsqueeze(0), speed=speed, low_hz=low_hz,
                                                       high_hz=high_hz, device='cuda:0')
                    variance_loss = sinc_loss.EMD_SSL(freqs, psd[i].unsqueeze(0), speed=speed, low_hz=low_hz,
                                                      high_hz=high_hz, device='cuda:0')
                    sparsity_loss = sinc_loss.SNR_SSL(freqs, psd[i].unsqueeze(0), speed=speed, low_hz=low_hz,
                                                      high_hz=high_hz, device='cuda:0')

                    entropy_ = bandwidth_loss + variance_loss + sparsity_loss
                    entropy.append(entropy_)

                entropy = torch.stack(entropy)

            # add into memory
            batch_data_ = batch_data[:-1].view(-1, 180, 3, 72, 72)

            for i, data in enumerate(batch_data_):
                p_l = int(pseudo_label[i].item())

                if p_l >= 8 or p_l < 0:
                    continue

                uncertainty = entropy[i].item()
                current_instance = (data, p_l, uncertainty)
                self.mem.add_instance(current_instance)
                self.current_instance += 1

                if self.current_instance % self.update_frequency == 0:
                    for i in range(1):
                        self.update_model(model, optimizer)

            # ema_out = self.model_ema(batch_data)

            return ema_out
    def update_model(self, model, optimizer):

        INPUT_3D = 1

        if INPUT_3D:
            model.train()
            self.model_ema.train()
            # get memory data
            sup_data, ages = self.mem.get_memory()
            l_sup = None

            initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}
            #kld_loss = torch.nn.KLDivLoss(reduction='none')
            kld_loss = torch.nn.KLDivLoss(reduction='batchmean')

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

                neg_auginfo = Auginfo(sup_data.shape[2])

                pos_auginfo = Auginfo(sup_data.shape[2])
                pos_auginfo.aug_speed = 0

                instance_weight = timeliness_reweighting(ages)
                l_sup_list =[]
                sup_out_lit = []
                ema_sup_out_list = []
                neg_stu_sup_out_list = []
                pos_stu_sup_out_list = []

                neg_speed_list = []
                pos_speed_list = []

                for idx, sup_data_ in enumerate(sup_data):
                    neg_sup_aug , speed = self.transform(neg_auginfo, sup_data_.cpu().numpy()) # [C,T,H,W]
                    neg_speed_list.append(speed)

                    pos_sup_aug , speed = self.transform(pos_auginfo, sup_data_.cpu().numpy()) # [C,T,H,W]
                    pos_speed_list.append(speed)

                    ema_sup_out,_,_,_ = self.model_ema(sup_data_.unsqueeze(0).cuda())
                    neg_stu_sup_out,_,_,_ = model(neg_sup_aug.unsqueeze(0).cuda())
                    pos_stu_sup_out,_,_,_ = model(pos_sup_aug.unsqueeze(0).cuda())

                    fps = float(30)
                    low_hz = float(0.66666667)
                    high_hz = float(3.0)

                    ema_sup_out = tent.torch_detrend(torch.cumsum(ema_sup_out.T, axis=0), torch.tensor(100.0))
                    neg_stu_sup_out = tent.torch_detrend(torch.cumsum(neg_stu_sup_out.T, axis=0), torch.tensor(100.0))
                    pos_stu_sup_out = tent.torch_detrend(torch.cumsum(pos_stu_sup_out.T, axis=0), torch.tensor(100.0))

                    ema_sup_out_list.append(ema_sup_out)
                    neg_stu_sup_out_list.append(neg_stu_sup_out)
                    pos_stu_sup_out_list.append(pos_stu_sup_out)

                    stu_sup_out,_,_,_ = model(sup_data_.unsqueeze(0).cuda())
                    stu_sup_out = tent.torch_detrend(torch.cumsum(stu_sup_out.T, axis=0), torch.tensor(100.0))
                    sup_out_lit.append(stu_sup_out)


                neg_speed_list = torch.tensor(neg_speed_list)
                pos_speed_list = torch.tensor(pos_speed_list)


                ema_sup_out_list = torch.stack(ema_sup_out_list)
                neg_stu_sup_out_list = torch.stack(neg_stu_sup_out_list)
                pos_stu_sup_out_list = torch.stack(pos_stu_sup_out_list)
                sup_out_lit = torch.stack(sup_out_lit)

                stu_freqs, stu_psd = tent.torch_power_spectral_density(sup_out_lit, fps=fps, low_hz=low_hz,
                                                                       high_hz=high_hz, normalize=False, bandpass=False)

                ema_freqs, ema_psd = tent.torch_power_spectral_density(ema_sup_out_list, fps=fps, low_hz=low_hz,
                                                                       high_hz=high_hz, normalize=False, bandpass=False)

                neg_stu_freqs, neg_stu_psd = tent.torch_power_spectral_density(neg_stu_sup_out_list, fps=fps, low_hz=low_hz,
                                                                       high_hz=high_hz,
                                                                       normalize=False, bandpass=False)

                pos_stu_freqs, pos_stu_psd = tent.torch_power_spectral_density(pos_stu_sup_out_list, fps=fps, low_hz=low_hz,
                                                                              high_hz=high_hz,
                                                                              normalize=False, bandpass=False)

                stu_psd = tent.normalize_psd(stu_psd)
                ema_psd = tent.normalize_psd(ema_psd)
                neg_stu_psd = tent.normalize_psd(neg_stu_psd)
                pos_stu_psd = tent.normalize_psd(pos_stu_psd)

                """
                criterion = torch.nn.CrossEntropyLoss()
                y_pred = batched_normed_psd(pos_stu_psd,neg_stu_psd)
                y_labels = label_distance(pos_speed_list, neg_speed_list)
                simper_loss = criterion (y_pred,y_labels.cuda())
                """


                """
                plt.plot(ema_psd[0].detach().cpu().numpy())
                plt.plot(neg_stu_psd[0].detach().cpu().numpy())
                plt.plot(pos_stu_psd[0].detach().cpu().numpy())
                plt.show()
                """

                bandwidth_loss = sinc_loss.IPR_SSL(stu_freqs, stu_psd, speed=pos_speed_list, low_hz=low_hz,
                                                   high_hz=high_hz, device='cuda:0')
                variance_loss = sinc_loss.EMD_SSL(stu_freqs, stu_psd, speed=pos_speed_list, low_hz=low_hz,
                                                  high_hz=high_hz, device='cuda:0')
                sparsity_loss = sinc_loss.SNR_SSL(stu_freqs, stu_psd, speed=pos_speed_list, low_hz=low_hz,
                                                  high_hz=high_hz, device='cuda:0')

                origin_sinc_loss_total = bandwidth_loss + variance_loss + sparsity_loss

                bandwidth_loss = sinc_loss.IPR_SSL(pos_stu_freqs, pos_stu_psd, speed=pos_speed_list, low_hz=low_hz,
                                                   high_hz=high_hz, device='cuda:0')
                variance_loss = sinc_loss.EMD_SSL(pos_stu_freqs, pos_stu_psd, speed=pos_speed_list, low_hz=low_hz,
                                                  high_hz=high_hz, device='cuda:0')
                sparsity_loss = sinc_loss.SNR_SSL(pos_stu_freqs, pos_stu_psd, speed=pos_speed_list, low_hz=low_hz,
                                                  high_hz=high_hz, device='cuda:0')

                pos_sinc_loss_total = bandwidth_loss + variance_loss + sparsity_loss

                bandwidth_loss = sinc_loss.IPR_SSL(neg_stu_freqs, neg_stu_psd, speed=neg_speed_list, low_hz=low_hz,
                                                   high_hz=high_hz, device='cuda:0')
                variance_loss = sinc_loss.EMD_SSL(neg_stu_freqs, neg_stu_psd, speed=neg_speed_list, low_hz=low_hz,
                                                  high_hz=high_hz, device='cuda:0')
                sparsity_loss = sinc_loss.SNR_SSL(neg_stu_freqs, neg_stu_psd, speed=neg_speed_list, low_hz=low_hz,
                                                  high_hz=high_hz, device='cuda:0')


                neg_sinc_loss_total = bandwidth_loss  + variance_loss + sparsity_loss

                sinc_loss_total = pos_sinc_loss_total + neg_sinc_loss_total + origin_sinc_loss_total
                #l_consistency = sum(abs(stu_sup_out - ema_sup_out) * instance_weight[idx])

                l_kld_consistency = kld_loss((pos_stu_psd+1e-10).log(), ema_psd)
                #l_kld_consistency = (kld_loss((pos_stu_psd + 1e-10).log(), ema_psd) * instance_weight).sum() / pos_stu_psd.size(0)

                #l_kld_consistency = (softmax_entropy(pos_stu_psd, ema_psd)*instance_weight).mean()

                print("origin_sinc_loss_total:", origin_sinc_loss_total)
                print ("pos_sinc_loss_total:", pos_sinc_loss_total)
                print ("neg_sinc_loss_total:", neg_sinc_loss_total)
                print ("l_kld_consistency:", l_kld_consistency)
                #print("simper_loss:", simper_loss)

        else:
            model.train()
            self.model_ema.train()
            # get memory data
            sup_data, ages = self.mem.get_memory()
            l_sup = None

            initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}
            # kld_loss = torch.nn.KLDivLoss(reduction='none')
            kld_loss = torch.nn.KLDivLoss(reduction='batchmean')

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

                neg_auginfo = Auginfo(sup_data.shape[1] + 1)

                pos_auginfo = Auginfo(sup_data.shape[1] + 1)
                pos_auginfo.aug_speed = 0

                instance_weight = timeliness_reweighting(ages)
                l_sup_list = []
                sup_out_lit = []
                ema_sup_out_list = []
                neg_stu_sup_out_list = []
                pos_stu_sup_out_list = []

                neg_speed_list = []
                pos_speed_list = []

                for idx, sup_data_ in enumerate(sup_data):
                    last_frame = torch.unsqueeze(sup_data_[-1, :, :, :], 0).repeat(num_of_gpu, 1, 1, 1)
                    sup_data_ = torch.cat((sup_data_, last_frame), 0)  # [T,C,H,W]
                    neg_sup_aug, speed = self.transform(neg_auginfo,
                                                        sup_data_.permute(0, 2, 3, 1).cpu().numpy())  # [C,T,H,W]
                    neg_speed_list.append(speed)

                    pos_sup_aug, speed = self.transform(pos_auginfo,
                                                        sup_data_.permute(1, 0, 2, 3).cpu().numpy())  # [C,T,H,W]
                    pos_speed_list.append(speed)

                    ema_sup_out = self.model_ema(sup_data_)
                    neg_stu_sup_out = model(neg_sup_aug.permute(1, 0, 2, 3))
                    pos_stu_sup_out = model(pos_sup_aug.permute(1, 0, 2, 3))

                    fps = float(30)
                    low_hz = float(0.66666667)
                    high_hz = float(3.0)

                    ema_sup_out = tent.torch_detrend(torch.cumsum(ema_sup_out, axis=0), torch.tensor(100.0))
                    neg_stu_sup_out = tent.torch_detrend(torch.cumsum(neg_stu_sup_out, axis=0), torch.tensor(100.0))
                    pos_stu_sup_out = tent.torch_detrend(torch.cumsum(pos_stu_sup_out, axis=0), torch.tensor(100.0))

                    ema_sup_out_list.append(ema_sup_out)
                    neg_stu_sup_out_list.append(neg_stu_sup_out)
                    pos_stu_sup_out_list.append(pos_stu_sup_out)

                    stu_sup_out = self.model_ema(sup_data_)
                    stu_sup_out = tent.torch_detrend(torch.cumsum(stu_sup_out, axis=0), torch.tensor(100.0))
                    sup_out_lit.append(stu_sup_out)

                neg_speed_list = torch.tensor(neg_speed_list)
                pos_speed_list = torch.tensor(pos_speed_list)

                ema_sup_out_list = torch.stack(ema_sup_out_list)
                neg_stu_sup_out_list = torch.stack(neg_stu_sup_out_list)
                pos_stu_sup_out_list = torch.stack(pos_stu_sup_out_list)
                sup_out_lit = torch.stack(sup_out_lit)

                stu_freqs, stu_psd = tent.torch_power_spectral_density(sup_out_lit, fps=fps, low_hz=low_hz,
                                                                       high_hz=high_hz, normalize=False, bandpass=False)

                ema_freqs, ema_psd = tent.torch_power_spectral_density(ema_sup_out_list, fps=fps, low_hz=low_hz,
                                                                       high_hz=high_hz, normalize=False, bandpass=False)

                neg_stu_freqs, neg_stu_psd = tent.torch_power_spectral_density(neg_stu_sup_out_list, fps=fps,
                                                                               low_hz=low_hz,
                                                                               high_hz=high_hz,
                                                                               normalize=False, bandpass=False)

                pos_stu_freqs, pos_stu_psd = tent.torch_power_spectral_density(pos_stu_sup_out_list, fps=fps,
                                                                               low_hz=low_hz,
                                                                               high_hz=high_hz,
                                                                               normalize=False, bandpass=False)

                stu_psd = tent.normalize_psd(stu_psd)
                ema_psd = tent.normalize_psd(ema_psd)
                neg_stu_psd = tent.normalize_psd(neg_stu_psd)
                pos_stu_psd = tent.normalize_psd(pos_stu_psd)

                """
                criterion = torch.nn.CrossEntropyLoss()
                y_pred = batched_normed_psd(pos_stu_psd,neg_stu_psd)
                y_labels = label_distance(pos_speed_list, neg_speed_list)
                simper_loss = criterion (y_pred,y_labels.cuda())
                """

                """
                plt.plot(ema_psd[0].detach().cpu().numpy())
                plt.plot(neg_stu_psd[0].detach().cpu().numpy())
                plt.plot(pos_stu_psd[0].detach().cpu().numpy())
                plt.show()
                """

                bandwidth_loss = sinc_loss.IPR_SSL(stu_freqs, stu_psd, speed=pos_speed_list, low_hz=low_hz,
                                                   high_hz=high_hz, device='cuda:0')
                variance_loss = sinc_loss.EMD_SSL(stu_freqs, stu_psd, speed=pos_speed_list, low_hz=low_hz,
                                                  high_hz=high_hz, device='cuda:0')
                sparsity_loss = sinc_loss.SNR_SSL(stu_freqs, stu_psd, speed=pos_speed_list, low_hz=low_hz,
                                                  high_hz=high_hz, device='cuda:0')

                origin_sinc_loss_total = bandwidth_loss + variance_loss + sparsity_loss

                bandwidth_loss = sinc_loss.IPR_SSL(pos_stu_freqs, pos_stu_psd, speed=pos_speed_list, low_hz=low_hz,
                                                   high_hz=high_hz, device='cuda:0')
                variance_loss = sinc_loss.EMD_SSL(pos_stu_freqs, pos_stu_psd, speed=pos_speed_list, low_hz=low_hz,
                                                  high_hz=high_hz, device='cuda:0')
                sparsity_loss = sinc_loss.SNR_SSL(pos_stu_freqs, pos_stu_psd, speed=pos_speed_list, low_hz=low_hz,
                                                  high_hz=high_hz, device='cuda:0')

                pos_sinc_loss_total = bandwidth_loss + variance_loss + sparsity_loss

                bandwidth_loss = sinc_loss.IPR_SSL(neg_stu_freqs, neg_stu_psd, speed=neg_speed_list, low_hz=low_hz,
                                                   high_hz=high_hz, device='cuda:0')
                variance_loss = sinc_loss.EMD_SSL(neg_stu_freqs, neg_stu_psd, speed=neg_speed_list, low_hz=low_hz,
                                                  high_hz=high_hz, device='cuda:0')
                sparsity_loss = sinc_loss.SNR_SSL(neg_stu_freqs, neg_stu_psd, speed=neg_speed_list, low_hz=low_hz,
                                                  high_hz=high_hz, device='cuda:0')

                neg_sinc_loss_total = bandwidth_loss + variance_loss + sparsity_loss

                sinc_loss_total = pos_sinc_loss_total + neg_sinc_loss_total + origin_sinc_loss_total
                # l_consistency = sum(abs(stu_sup_out - ema_sup_out) * instance_weight[idx])

                l_kld_consistency = kld_loss((pos_stu_psd + 1e-10).log(), ema_psd)
                # l_kld_consistency = (kld_loss((pos_stu_psd + 1e-10).log(), ema_psd) * instance_weight).sum() / pos_stu_psd.size(0)

                # l_kld_consistency = (softmax_entropy(pos_stu_psd, ema_psd)*instance_weight).mean()

                print("origin_sinc_loss_total:", origin_sinc_loss_total)
                print("pos_sinc_loss_total:", pos_sinc_loss_total)
                print("neg_sinc_loss_total:", neg_sinc_loss_total)
                print("l_kld_consistency:", l_kld_consistency)
                # print("simper_loss:", simper_loss)


        l = sinc_loss_total  + l_kld_consistency  #+ simper_loss
        #l = l_kld_consistency
        print(l)
        if l is not None:
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        """
        for name, param in model.named_parameters():
            if has_parameter_changed(initial_params[name], param):
                print(f"Parameter '{name}' has changed during training.")
            else:
                print(f"Parameter '{name}' remains unchanged.")
        """
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

            print (name)
            """
            if name.find('motion_conv')!=-1:
                target_layer = get_named_submodule(model, name)
                target_layer.requires_grad_(True)
                
            """
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d) \
                    or isinstance(sub_module, nn.BatchNorm3d):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = RobustBN2d
            elif isinstance(bn_layer, nn.BatchNorm3d):
                NewBN = RobustBN3d
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

def has_parameter_changed(initial, current):
    return not torch.all(torch.eq(initial, current))
