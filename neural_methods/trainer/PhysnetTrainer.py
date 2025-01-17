"""PhysNet Trainer."""
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm

import neural_methods.trainer.tent as tent
from neural_methods.adapter import build_adapter
import neural_methods.loss.sinc_loss as sinc_loss
import neural_methods.trainer.tent as tent
import neural_methods.augmentation.sinc_aug as sinc_aug

import matplotlib.pyplot as plt
import gc
import wandb

"""PhysNet Trainer."""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import (PhysNet_padding_Encoder_Decoder_MAX, PhysNet_padding_Encoder_Decoder_MAX_color
,PhysNet_padding_Encoder_Decoder_MAX_1x1conv)
from neural_methods.model.PhysNet_def import PhysNet_padding_Encoder_Decoder_MAX_def
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm

import neural_methods.trainer.tent as tent
from neural_methods.adapter import build_adapter
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize

from tent import get_filtered_freqs_psd

DEBUG = 0
SAVE = 0
SAVE_WEIGHTS = 0

class PhysnetTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        if config.MODEL.NAME =='Physnet_def':
            self.model = PhysNet_padding_Encoder_Decoder_MAX_def(
                frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)
        elif config.MODEL.NAME =='Physnet_color':
            self.model = PhysNet_padding_Encoder_Decoder_MAX_color(
                frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)
        elif config.MODEL.NAME =='Physnet_1x1conv':
            self.model = PhysNet_padding_Encoder_Decoder_MAX_1x1conv(
                frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)
        else:
            self.model = PhysNet_padding_Encoder_Decoder_MAX(
                frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_model = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

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
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                rPPG, x_visual, x_visual3232, x_visual1616, features_before_pooling = self.model(
                    batch[0].to(torch.float32).to(self.device))
                BVP_label = batch[1].to(
                    torch.float32).to(self.device)
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss = self.loss_model(rPPG, BVP_label)
                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
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
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(
                    torch.float32).to(self.device)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    valid_batch[0].to(torch.float32).to(self.device))
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss_ecg = self.loss_model(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                valid_step += 1
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")

            # Load the state_dict from the saved model
            saved_state_dict = torch.load(self.config.INFERENCE.MODEL_PATH)
            # Create a new state_dict without the 'model.' prefix
            new_state_dict = {k.replace("model.", ""): v for k, v in saved_state_dict.items()}
            # Load the new state_dict into the model
            incompatible_keys = self.model.load_state_dict(new_state_dict, strict=False)

            print(f"Missing keys: {incompatible_keys.missing_keys}")
            print(f"Unexpected keys: {incompatible_keys.unexpected_keys}")

            #self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
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

        hr_log = []
        feature_log = []
        subj_log =[]

        with torch.no_grad():
            for idx, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                pred_ppg_test, x_visual, x_visual3232, x_visual1616 , features_before_pooling = self.model(data)

                subj = test_batch[2]

                if SAVE:
                    print (x_visual1616.shape)

                    predictions_np = pred_ppg_test.detach().cpu().numpy()

                    for idx,predictions_ in  enumerate(predictions_np):
                        #predictions_ = (predictions_ - np.mean(predictions_)) / np.std(predictions_)
                        #freqs, psd, hr_p = tent.get_filtered_freqs_psd(predictions_)

                        freqs, psd, hr_p = tent.get_filtered_freqs_psd(label.detach().cpu().numpy())
                        hr_log.append(hr_p)
                        feature_log.append(x_visual1616[idx].detach().cpu().numpy())
                        subj_log.append(list(subj)[idx])
                        print (hr_p, list(subj)[idx])

                if DEBUG:
                    ax1 = plt.subplot(2, 1, 1)

                    predictions_np = pred_ppg_test[0].detach().cpu().numpy()

                    predictions_np = (predictions_np - np.mean(predictions_np)) / np.std(predictions_np)
                    label_np = label[0].detach().cpu().numpy()

                    ax1.plot(predictions_np, 'blue')
                    ax1.plot(label_np, 'red')

                    ax2 = plt.subplot(2, 1, 2)

                    freqs, psd, hr_p = tent.get_filtered_freqs_psd(predictions_np)
                    freqs, y_psd, hr_r = tent.get_filtered_freqs_psd(label_np)

                    error = abs(hr_p - hr_r)
                    # error = abs(freqs[y_psd[0].argmax(dim=0)] - freqs[psd[0].argmax(dim=0)]).detach().cpu().numpy()

                    ax2.plot(psd[0].detach().cpu().numpy(), color='blue')
                    ax2.plot(y_psd[0].detach().cpu().numpy(), color='red')
                    ax2.axvline(x=psd[0].detach().cpu().numpy().argmax(), color='blue', linestyle='--')
                    ax2.axvline(x=y_psd[0].detach().cpu().numpy().argmax(), color='red', linestyle='--')

                    ax2.axvline(x=120, color='gray', linestyle='-')
                    ax2.axvline(x=540, color='gray', linestyle='-')

                    plt.suptitle('Various Straight Lines', fontsize=20)

                    title = f"iter_count:{idx},error:{round(float(error), 2)},"
                    plt.suptitle(title, fontsize=20)

                    plt.show()
                    plt.close()
                    plt.clf()


                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]


        if SAVE:

            norm = Normalize(vmin=min(hr_log), vmax=max(hr_log))
            cmap = ListedColormap(plt.cm.jet(np.linspace(0, 1, len(hr_log))))

            features = np.array(feature_log)
            flattened_features = np.reshape(features, (features.shape[0] , -1))

            tsne = TSNE(n_components=2)  # You can change n_components to 3 for 3D visualization
            embedded_features = tsne.fit_transform(flattened_features)


            sub_list = []
            for sub in subj_log:
                sub_list.append(sub.split("_")[0])

            unique_subjs = list(set(sub_list))

            print ("unique_subjs:",len(unique_subjs))
            edge_colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(unique_subjs)))  # You can choose a different colormap

            for idx, hr in enumerate(hr_log):
                hr_color = cmap(norm(hr))
                subj = subj_log[idx]
                subj_color = edge_colors[unique_subjs.index(subj)]

                if subj.find("L3")!=-1:
                    marker_shape ='o'
                else:
                    marker_shape = '*'

                plt.scatter(embedded_features[idx, 0], embedded_features[idx, 1], c=hr_color,linewidths=2,
                            cmap="gist_rainbow", marker=marker_shape)
                #plt.scatter(embedded_features[idx, 0], embedded_features[idx, 1], c=color,
                #            edgecolors=edge_color, linewidths=2, cmap="gist_rainbow")

            #scatter = plt.scatter(embedded_features[:, 0], embedded_features[:, 1],c=hr_log, cmap=cmap)
            #plt.colorbar(scatter, label='hr_log')
            plt.colorbar(label='hr_log')
            plt.title('t-SNE Visualization')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.show()

            plt.hist(hr_log,bins=100)
            plt.show()


        print('')
        calculate_metrics(predictions, labels, self.config)

    def save_model(self, index):

        # 기존 state_dict 불러오기
        original_state_dict = self.model.state_dict()

        # 새로운 state_dict 초기화
        new_state_dict = {}

        # 키에서 'model.' 접두어가 있는지 확인하고 제거
        for k, v in original_state_dict.items():
            if k.startswith('model.'):
                name = k[6:]  # 'model.' 접두어 제거
            else:
                name = k
            new_state_dict[name] = v


        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(new_state_dict, model_path)
        print('Saved Model Path: ', model_path)

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

    def setup_tent(self, model):
        """Set up tent adaptation.

        Configure the model for training + feature modulation by batch statistics,
        collect the parameters for feature modulation by gradient optimization,
        set up the optimizer, and then tent the model.
        """
        model = tent.configure_model(model,self.config)
        params, param_names = tent.collect_params(model)

        if self.config.MODEL.NAME == 'Physnet_color':
            model.scale_factor.requires_grad = True
            params.append(model.scale_factor)
            model.bias_factor.requires_grad = True
            params.append(model.bias_factor)

        optimizer = self.setup_optimizer(params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=2)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


        tent_model = tent.Tent(model, optimizer,scheduler,
                               steps=self.config.OPTIM.STEPS,
                               episodic=self.config.MODEL.EPISODIC,
                               model_name=self.config.MODEL.NAME,
                               b_scale=self.config.ADAPTER.TENT.B_SCALE,
                               s_scale=self.config.ADAPTER.TENT.S_SCALE,
                               v_scale=self.config.ADAPTER.TENT.V_SCALE,
                               sm_scale=self.config.ADAPTER.TENT.SM_SCALE,
                               fc_scale=self.config.ADAPTER.TENT.FC_SCALE)
        # logger.info(f"model for adaptation: %s", model)
        # logger.info(f"params for adaptation: %s", param_names)
        # logger.info(f"optimizer for adaptation: %s", optimizer)
        return tent_model

    def tta_tent(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===TTA===")
        predictions = dict()
        labels = dict()

        if self.config.INFERENCE.MODEL_PATH and not os.path.exists(self.config.INFERENCE.MODEL_PATH):
            raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")

        if self.config.INFERENCE.MODEL_PATH:
            state_dict = torch.load(self.config.INFERENCE.MODEL_PATH)
            #state_dict.pop("scale_factor", None)
            #state_dict.pop("bias_factor", None)

            incompatible_keys = self.model.load_state_dict(state_dict,strict=False)

            print(f"Missing keys: {incompatible_keys.missing_keys}")
            print(f"Unexpected keys: {incompatible_keys.unexpected_keys}")

            print("Testing uses pretrained model!")

        self.model = self.model.to(self.config.DEVICE)
        self.model = self.setup_tent(self.model)

        hr_log = []
        feature_log = []
        subj_log =[]
        current_subj = 0
        last_subj = 0

        #self.model.eval()
        with torch.no_grad():
            for idx, test_batch in enumerate(data_loader['test']):
                print(idx)
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                pred_ppg_test, x_visual, x_visual3232, x_visual1616, features_before_pooling = self.model(data,label)
                pred_ppg_test = pred_ppg_test.detach()

                subj = test_batch[2]
                current_subj = subj[-1]
                print (current_subj,last_subj)


                # debug label #
                """
                pred_ppg_test =( pred_ppg_test - pred_ppg_test.mean()) / pred_ppg_test.std()
                label = (label - label.mean()) / label.std()

                for idx in range(data.shape[0]):
                    freqs, psd_predict, hr_predict = tent.get_filtered_freqs_psd(pred_ppg_test.cpu().numpy()[idx])
                    freqs, psd_label, hr_label = tent.get_filtered_freqs_psd(label.cpu().numpy()[idx])

                    label_error = abs(hr_predict-hr_label)
                    plt.title(str(label_error))
                    plt.plot(pred_ppg_test.cpu().numpy()[idx])
                    plt.plot(label.detach().cpu().numpy()[idx])
                    #label = tent._detrend(np.cumsum(label.detach().cpu().numpy()), 100)
                    plt.show()
                """
                #if current_subj != last_subj:
                #    self.model.reset()

                last_subj = current_subj

                if SAVE:

                    # print (x_visual1616.shape)
                    predictions_np = pred_ppg_test.detach().cpu().numpy()
                    batch_log = []
                    for idx, predictions_ in enumerate(predictions_np):
                        #predictions_ = (predictions_ - np.mean(predictions_)) / np.std(predictions_)
                        freqs, psd, hr_p = tent.get_filtered_freqs_psd(predictions_)
                        #freqs, psd, hr_p = tent.get_filtered_freqs_psd(label.detach().cpu().numpy())
                        print(hr_p)
                        hr_log.append(hr_p)
                        feature_log.append(x_visual3232[idx].detach().cpu().numpy())
                        batch_log.append(x_visual3232[idx].detach().cpu().numpy())
                        subj_log.append(list(subj)[idx])
                        #print(hr_p, list(subj)[idx])

                    batch_log = np.array(batch_log)
                    flattened_features = [feature.flatten() for feature in batch_log]

                    # 거리를 저장할 변수 초기화
                    max_distance = 0
                    pair = (0, 0)

                    # 평균 거리를 저장할 변수 초기화
                    total_distance = 0
                    count = 0

                    # 연속된 feature 쌍에 대한 거리 계산
                    for i in range(len(flattened_features) - 1):
                        j = i + 1
                        distance = np.linalg.norm(flattened_features[i] - flattened_features[j])
                        total_distance += distance
                        count += 1
                        if distance > max_distance:
                            max_distance = distance
                            pair = (i, j)

                    # 평균 거리 계산
                    average_distance = total_distance / count
                    # 평균 거리 대비 max_distance 계산
                    relative_max_distance = max_distance / average_distance

                    print(f"The maximum distance is {relative_max_distance} between feature {pair[0]} and feature {pair[1]}")

                if DEBUG:
                    ax1 = plt.subplot(2, 1, 1)

                    predictions_np = pred_ppg_test[0].detach().cpu().numpy()

                    predictions_np = (predictions_np - np.mean(predictions_np)) / np.std(predictions_np)
                    label_np = label[0].detach().cpu().numpy()

                    ax1.plot(predictions_np, 'blue')
                    ax1.plot(label_np, 'red')

                    ax2 = plt.subplot(2, 1, 2)

                    freqs, psd , hr_p= tent.get_filtered_freqs_psd(predictions_np)
                    freqs, y_psd , hr_r= tent.get_filtered_freqs_psd(label_np)

                    error = abs(hr_p-hr_r)
                    #error = abs(freqs[y_psd[0].argmax(dim=0)] - freqs[psd[0].argmax(dim=0)]).detach().cpu().numpy()

                    ax2.plot(psd[0].detach().cpu().numpy(), color='blue')
                    ax2.plot(y_psd[0].detach().cpu().numpy(), color='red')
                    ax2.axvline(x=psd[0].detach().cpu().numpy().argmax(), color='blue', linestyle='--')
                    ax2.axvline(x=y_psd[0].detach().cpu().numpy().argmax(), color='red', linestyle='--')

                    ax2.axvline(x=120, color='gray', linestyle='-')
                    ax2.axvline(x=540, color='gray', linestyle='-')

                    plt.suptitle('Various Straight Lines', fontsize=20)

                    title = f"iter_count:{idx},error:{round(float(error), 2)},"
                    plt.suptitle(title, fontsize=20)

                    plt.show()
                    plt.close()
                    plt.clf()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]


        print('')
        calculate_metrics(predictions, labels, self.config)
        print("total_foward_count:",self.model.tta_fw_cnt)
        print("total_backward_count:",self.model.tta_bw_cnt)

        if SAVE_WEIGHTS:
            self.save_model(self.config.OPTIM.STEPS)

        if SAVE:

            norm = Normalize(vmin=min(hr_log), vmax=max(hr_log))

            cmap = ListedColormap(plt.cm.jet(np.linspace(0, 1, len(hr_log))))

            features = np.array(feature_log)
            flattened_features = np.reshape(features, (features.shape[0] , -1))

            tsne = TSNE(n_components=2)  # You can change n_components to 3 for 3D visualization
            embedded_features = tsne.fit_transform(flattened_features)

            sub_list = []
            for sub in subj_log:
                sub_list.append(sub.split("_")[0])

            unique_subjs = list(set(sub_list))

            print("unique_subjs:", len(unique_subjs))
            edge_colors = plt.cm.gist_rainbow(
                np.linspace(0, 1, len(unique_subjs)))  # You can choose a different colormap

            for idx, hr in enumerate(hr_log):
                hr_color = cmap(norm(hr))
                subj_info = subj_log[idx]
                subj = subj_info.split("_")[0]

                subj_color = edge_colors[unique_subjs.index(subj)]

                if subj_info.find("L3")!=-1:
                    marker_shape ='o'
                else:
                    marker_shape = '*'

                plt.scatter(embedded_features[idx, 0], embedded_features[idx, 1], c=hr_color,linewidths=2,
                            cmap="gist_rainbow", marker=marker_shape)

                # plt.scatter(embedded_features[idx, 0], embedded_features[idx, 1], c=color,
                #            edgecolors=edge_color, linewidths=2, cmap="gist_rainbow")

            # scatter = plt.scatter(embedded_features[:, 0], embedded_features[:, 1],c=hr_log, cmap=cmap)
            # plt.colorbar(scatter, label='hr_log')
            #plt.colorbar(label='hr_log')
            plt.title('t-SNE Visualization')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.show()

            plt.hist(hr_log, bins=100)
            plt.show()

        if DEBUG:
            freq_error_list = np.array(self.model.freq_error_list)
            total_loss_list = np.array(self.model.total_loss_list)

            data = {
            'freq_error':freq_error_list,
            'sinc_loss':total_loss_list}
            df =pd.DataFrame(data)
            scatter_data = wandb.Table(dataframe=df, allow_mixed_types=True)
            scatter_plot = wandb.plot.scatter(scatter_data, x='sinc_loss', y='freq_error', title='sincXfreq')
            wandb.log({"scatter_plot_all": scatter_plot})

            data = {
                'freq_error(mean)': freq_error_list.mean(),
                'sinc_loss(mean)': total_loss_list.mean()}
            df2 = pd.DataFrame(data,index = [0])
            scatter_data2 = wandb.Table(dataframe=df2, allow_mixed_types=True)
            scatter_plot2 = wandb.plot.scatter(scatter_data2, x='sinc_loss(mean)', y='freq_error(mean)', title='sincXfreq(mean)')
            wandb.log({"scatter_plot_mean": scatter_plot2})



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
            for idx, test_batch in enumerate(data_loader['test']):
                print (idx)
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                pred_ppg_test = self.model(data)

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]


        print('')
        calculate_metrics(predictions, labels, self.config)


    def train_ssl(self,data_loader):
        if os.path.exists(self.config.INFERENCE.MODEL_PATH):
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Traing SSL uses pretrained model!")

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                #predictions_batch, x_visual, x_visual3232, x_visual1616 = self.model(
                #    batch[0].to(torch.float32).to(self.device))
                #BVP_label = batch[1].to(
                #    torch.float32).to(self.device)
                #rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                #BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                #            torch.std(BVP_label)  # normalize
                #loss = self.loss_model(rPPG, BVP_label)

                fps = float(30)
                low_hz = float(0.66666667)
                high_hz = float(3.0)


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

                #print (batch[0].shape) # [4, 3, 128, 72, 72]
                total_loss = 0.0
                auginfo = Auginfo(batch[0].shape[2])

                prediction_list = []
                speed_list = []

                for data in batch[0]:

                    if self.config.AUG :
                        x_aug, speed = sinc_aug.apply_transformations(auginfo, data.cpu().numpy())  # [C,T,H,W]
                        predictions, _, _, _  = self.model(x_aug.unsqueeze(0).cuda())
                    else:
                        speed = 1.0
                        predictions, _, _, _ = self.model(data.unsqueeze(0).cuda())
                    predictions = (predictions - torch.mean(predictions)) / torch.std(predictions)  # normalize
                    predictions_smooth = tent.torch_detrend(torch.cumsum(predictions.T, axis=0), torch.tensor(100.0))
                    prediction_list.append(predictions_smooth)
                    speed_list.append(speed)

                prediction_list = torch.stack(prediction_list)
                predictions_batch = prediction_list.view(-1, 128)
                speed = torch.tensor(speed_list)


                freqs, psd = tent.torch_power_spectral_density(predictions_batch, fps=fps, low_hz=low_hz, high_hz=high_hz,
                                                      normalize=False, bandpass=False)

                bandwidth_loss = sinc_loss.IPR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
                variance_loss = sinc_loss.EMD_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')
                sparsity_loss = sinc_loss.SNR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device='cuda:0')

                loss = bandwidth_loss+ variance_loss +sparsity_loss


                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
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
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))

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