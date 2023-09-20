"""Trainer for EfficientPhys."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.EfficientPhys import EfficientPhys,EfficientPhys_color
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm

import neural_methods.trainer.tent as tent
import neural_methods.trainer.norm as norm

from neural_methods.adapter import build_adapter
import neural_methods.augmentation.sinc_aug as sinc_aug
import neural_methods.loss.sinc_loss as sinc_loss

class EfficientPhysTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.frame_depth = config.MODEL.EFFICIENTPHYS.FRAME_DEPTH
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu * self.frame_depth
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        
        if config.TOOLBOX_MODE == "train_and_test":
            self.model = EfficientPhys(frame_depth=self.frame_depth, img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H).to(
                self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            self.num_train_batches = len(data_loader["train"])
            self.criterion = torch.nn.MSELoss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":

            if config.MODEL.NAME == 'EfficientPhys_color':
                self.model = (EfficientPhys_color(frame_depth=self.frame_depth, img_size=config.TEST.DATA.PREPROCESS.RESIZE.H).
                to(self.device))
            else:
                self.model = (EfficientPhys(frame_depth=self.frame_depth, img_size=config.TEST.DATA.PREPROCESS.RESIZE.H).
                    to(self.device))
            #self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("EfficientPhys trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                data = data[:(N * D) // self.base_len * self.base_len]
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                data = torch.cat((data, last_frame), 0)
                labels = labels[:(N * D) // self.base_len * self.base_len]
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item())
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                data_valid = data_valid[:(N * D) // self.base_len * self.base_len]
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data_valid[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                data_valid = torch.cat((data_valid, last_frame), 0)
                labels_valid = labels_valid[:(N * D) // self.base_len * self.base_len]
                pred_ppg_valid = self.model(data_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data_test[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                data_test = torch.cat((data_test, last_frame), 0)
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                pred_ppg_test = self.model(data_test)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        print('')
        calculate_metrics(predictions, labels, self.config)

    def setup_optimizer(self, params):
        """Set up optimizer for tent adaptation.

        Tent needs an optimizer for test-time entropy minimization.
        In principle, tent could make use of any gradient optimizer.
        In practice, we advise choosing Adam or SGD+momentum.
        For optimization settings, we advise to use the settings from the end of
        trainig, if known, or start with a low learning rate (like 0.001) if not.

        For best results, try tuning the learning rate and batch size.
        """
        if self.config.OPTIM.METHOD == 'Adam':
            return optim.Adam(params,
                              lr=self.config.OPTIM.LR,
                              betas=(self.config.OPTIM.BETA, 0.999),
                              weight_decay=self.config.OPTIM.WD)
        elif self.config.OPTIM.METHOD == 'SGD':
            return optim.SGD(params,
                             lr=self.config.OPTIM.LR,
                             momentum=self.config.OPTIM.MOMENTUM,
                             dampening=self.config.OPTIM.DAMPENING,
                             weight_decay=self.config.OPTIM.WD,
                             nesterov=self.config.OPTIM.NESTEROV)
        else:
            raise NotImplementedError

    def setup_norm(self, model):
        """Set up test-time normalization adaptation.

        Adapt by normalizing features with test batch statistics.
        The statistics are measured independently for each batch;
        no running average or other cross-batch estimation is used.
        """
        norm_model = norm.Norm(model)
        #logger.info(f"model for adaptation: %s", model)
        stats, stat_names = norm.collect_stats(model)
        #logger.info(f"stats for adaptation: %s", stat_names)
        return norm_model

    def setup_tent(self, model):
        """Set up tent adaptation.

        Configure the model for training + feature modulation by batch statistics,
        collect the parameters for feature modulation by gradient optimization,
        set up the optimizer, and then tent the model.
        """
        model = tent.configure_model(model,self.config)
        params, param_names = tent.collect_params(model)


        if self.config.MODEL.NAME == 'EfficientPhys_color':
            model.scale_factor.requires_grad = True
            params.append(model.scale_factor)
            model.bias_factor.requires_grad = True
            params.append(model.bias_factor)

        optimizer = self.setup_optimizer(params)
        tent_model = tent.Tent(model, optimizer,
                               steps=self.config.OPTIM.STEPS,
                               episodic=self.config.MODEL.EPISODIC,
                               model_name = self.config.MODEL.NAME)
        #logger.info(f"model for adaptation: %s", model)
        #logger.info(f"params for adaptation: %s", param_names)
        #logger.info(f"optimizer for adaptation: %s", optimizer)
        return tent_model

    def tta_tent(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===TTA===")
        predictions = dict()
        labels = dict()

        if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
            raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")

        state_dict = torch.load(self.config.INFERENCE.MODEL_PATH)
        #state_dict.pop("scale_factor", None)
        #state_dict.pop("bias_factor", None)
        #self.model.load_state_dict(state_dict,strict=False)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of DataParallel
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict,strict=False)

        #self.model.load_state_dict(state_dict)
        print("Testing uses pretrained model!")

        self.model = self.model.to(self.config.DEVICE)
        self.model = self.setup_tent(self.model)

        #self.model.eval()
        with torch.no_grad():
            for idx, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                print (idx, batch_size)
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data_test[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                data_test = torch.cat((data_test, last_frame), 0)
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                pred_ppg_test = self.model(data_test,labels_test).detach()
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        print('')
        calculate_metrics(predictions, labels, self.config)

    def tta_rotta(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===TTA===")
        predictions = dict()
        labels = dict()

        if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
            raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
        self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
        print("Testing uses pretrained model!")

        self.model = self.model.to(self.config.DEVICE)
        optimizer = build_optimizer(self.config)
        tta_adapter = build_adapter(self.config)
        self.model = tta_adapter(self.config, self.model, optimizer).cuda()

        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data_test[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                data_test = torch.cat((data_test, last_frame), 0)
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                pred_ppg_test = self.model(data_test)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        print('')
        calculate_metrics(predictions, labels, self.config)

    def train_ssl(self, data_loader):

        if os.path.exists(self.config.INFERENCE.MODEL_PATH):
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Traing SSL uses pretrained model!")


        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                data = data[:(N * D) // self.base_len * self.base_len]
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                data = torch.cat((data, last_frame), 0) # [T,C,H,W]
                labels = labels[:(N * D) // self.base_len * self.base_len]
                self.optimizer.zero_grad()
                #pred_ppg = self.model(data)
                #loss = self.criterion(pred_ppg, labels)

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


                auginfo = Auginfo(data.shape[0])

                fps = float(30)
                low_hz = float(0.66666667)
                high_hz = float(3.0)

                # [T,C,H,W]
                x_aug, speed = sinc_aug.apply_transformations(auginfo,
                                                              data.permute(0, 2, 3, 1).cpu().numpy())  # [C,H,W,C]
                predictions = self.model(x_aug.permute(1, 0, 2, 3))

                predictions_smooth = tent.torch_detrend(torch.cumsum(predictions, axis=0), torch.tensor(100.0))

                predictions_batch = predictions_smooth.view(-1,180)
                #predictions_batch = predictions_smooth.T


                freqs, psd = tent.torch_power_spectral_density(predictions_batch, fps=fps, low_hz=low_hz,
                                                          high_hz=high_hz,
                                                          normalize=False, bandpass=False)

                speed = torch.tensor([speed]*16)

                bandwidth_loss = sinc_loss.IPR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz,
                                                   device='cuda:0')
                variance_loss = sinc_loss.EMD_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz,
                                                  device='cuda:0')
                sparsity_loss = sinc_loss.SNR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz,
                                                  device='cuda:0')
                #print("bandwidth_loss:", bandwidth_loss, "sparsity_loss:", sparsity_loss, "variance_loss:",
                #      variance_loss)

                loss = bandwidth_loss + variance_loss + sparsity_loss

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item())
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))


    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)


def build_optimizer(cfg):
    def optimizer(params):
        if cfg.OPTIM.METHOD == 'Adam':
            return optim.Adam(params,
                              lr=cfg.OPTIM.LR,
                              betas=(cfg.OPTIM.BETA, 0.999),
                              weight_decay=cfg.OPTIM.WD)
        elif cfg.OPTIM.METHOD == 'SGD':
            return optim.SGD(params,
                             lr=cfg.OPTIM.LR,
                             momentum=cfg.OPTIM.MOMENTUM,
                             dampening=cfg.OPTIM.DAMPENING,
                             weight_decay=cfg.OPTIM.WD,
                             nesterov=cfg.OPTIM.NESTEROV)
        else:
            raise NotImplementedError

    return optimizer