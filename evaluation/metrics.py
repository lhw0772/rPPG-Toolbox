import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
from tqdm import tqdm

import matplotlib.pyplot as plt

import sys
def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr_peak, pred_hr_peak, SNR = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
                gt_hr_peak_all.append(gt_hr_peak)
                predict_hr_peak_all.append(pred_hr_peak)
                SNR_all.append(SNR)
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                SNR_all.append(SNR)
            else:
                raise ValueError("Inference evaluation method name wrong!")

    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        predict_hr_fft_all = np.array(predict_hr_fft_all)

        if config.LOG.FILE == False:
            # Your existing data and plotting code
            plt.subplot(2, 1, 1)
            plt.plot(gt_hr_fft_all, color='red')
            plt.plot(predict_hr_fft_all, color='blue')
            plt.plot(abs(gt_hr_fft_all - predict_hr_fft_all), color='green')

            plt.subplot(2, 1, 2)
            plt.scatter(x=predict_hr_fft_all, y=gt_hr_fft_all)

            # Original correlation line
            fit = np.polyfit(predict_hr_fft_all, gt_hr_fft_all, 1)
            fit_fn = np.poly1d(fit)
            plt.plot(np.unique(predict_hr_fft_all), fit_fn(np.unique(predict_hr_fft_all)), color='red',
                     label='Correlation Line')

            # Upper and lower boundaries
            upper_boundary = fit_fn(np.unique(predict_hr_fft_all)) + 5
            lower_boundary = fit_fn(np.unique(predict_hr_fft_all)) - 5

            # Plotting the boundaries
            plt.plot(np.unique(predict_hr_fft_all), upper_boundary, '--', color='red', label='+5 bpm')
            plt.plot(np.unique(predict_hr_fft_all), lower_boundary, '--', color='red', label='-5 bpm')

            #plt.legend()
            plt.show()
        else:
            sys.stdout = open(f'log/{config.TEST.DATA.INFO}_{config.ADAPTER.TENT}.txt', 'w')

        SNR_all = np.array(SNR_all)
        num_test_samples = len(predict_hr_fft_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)

                MAE_FFT = round(MAE_FFT, 2)
                standard_error = round(standard_error, 2)

                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
            elif metric == "RMSE":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                standard_error = np.std(np.square(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)

                RMSE_FFT = round(RMSE_FFT, 2)
                standard_error = round(standard_error, 2)

                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100

                MAPE_FFT = round(MAPE_FFT, 2)
                standard_error = round(standard_error, 2)


                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))

                correlation_coefficient = round(correlation_coefficient, 2)
                standard_error = round(standard_error, 2)

                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)

                SNR_FFT = round(SNR_FFT,2)
                standard_error = round(standard_error, 2)

                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
            elif "AU" in metric:
                pass
            else:
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        SNR_all = np.array(SNR_all)
        num_test_samples = len(predict_hr_peak_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                standard_error = np.std(np.square(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
            elif "AU" in metric:
                pass
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")
