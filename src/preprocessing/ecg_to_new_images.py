import os
import wfdb
import numpy
import matplotlib.pyplot as plt
from biosppy.signals import ecg
from ecgdetectors import Detectors
from scipy.interpolate import interp1d

from denoise import denoise_signal
from baseline_wander import remove_baseline_wander

# record_db = MIT-BIH database
record_db = "../../MIT-BIH-DB"

# split all files ending in .dat, returns list of filenames (001, 002 etc)
records = [f.split('.')[0] for f in os.listdir(record_db) if f.endswith('.dat')]

sampling_rate = 360
detectors = Detectors(sampling_rate)

# Loop through each patient_nums record. Each record has a set of 3 files:
# 001.dat, 001.hea, 001.atr, which are contained inside the locally stored MIT-BIH-DB
for patient_num in records:
    print(f"Processing record: {patient_num}")

    record_data = wfdb.rdsamp(f'../../MIT-BIH-DB/{patient_num}')
    annotation = wfdb.rdann(f'../../MIT-BIH-DB/{patient_num}', 'atr')

    # Get the ECG values from the file and choose the ecg_data to process 
    ecg_data = record_data[0].transpose()
    ecg_data = ecg_data[0]  # This will be MLII for all records but 102, 104, 114 which will be V5

    # Denoise the data
    ecg_data = denoise_signal(ecg_data)

    # Remove baseline wander
    ecg_data = remove_baseline_wander(ecg_data)

    # method with biospyy.signals ecg
    rpeaks_indices = ecg.ecg(signal=ecg_data, sampling_rate=360, show=False) #identify ecg features, including r peak

    # used for how much of the ECG data will be contained in an image
    pre_R_window = 128  # Number of samples before the R-peak
    post_R_window = 128  # Number of samples after the R-peak

    for idx, rpeak_idx in enumerate(rpeaks_indices['rpeaks']):
        # Find the annotation closest to the R-peak
        closest_annotation_idx = numpy.argmin(numpy.abs(annotation.sample - rpeak_idx))
        closest_annotation = annotation.symbol[closest_annotation_idx]

        # Define the range for the current beat (centered around the R-peak)
        start_idx = max(rpeak_idx - pre_R_window, 0)
        end_idx = min(rpeak_idx + post_R_window, len(ecg_data))

        # Extract the heartbeat segment
        beat = ecg_data[start_idx:end_idx]

        # Plot and save the original signal
        #if closest_annotation != 'N':
        #    figO, axo = plt.subplots(figsize=(1.66, 1.38), dpi=100)
        #    axo.plot(beat, color='black')
        #    axo.set_title(f'Original {closest_annotation}')
        #    axo.set_xlim(0, len(beat))
        #    axo.axis('off')
        #    figO.savefig(f'../../created-images/Edited/{patient_num}_{idx}.png', bbox_inches='tight', pad_inches=0)
        #    plt.close(figO)

        # Beat stretch
        #x_original = numpy.arange(len(beat)) - 128
        #mag = 1.05
        #start = -128/mag
        #stop = 127/mag   # numpy polynomial 0.9 - 1.1 times by ecg data = new beat
        #x_stretched = numpy.linspace(start, stop, 256)  # Stretch by 10%
        #interpolator = interp1d(x_original, beat, kind='quadratic')
        #beat_stretched = interpolator(x_stretched)

        # Beat stretch
        x_original = numpy.arange(len(beat))
        x_stretched = numpy.linspace(0, len(beat) - 1, int(len(beat) * 1.5))  
        interpolator = interp1d(x_original, beat, kind='linear')
        beat_stretched = interpolator(x_stretched)

        middle_idx = len(beat_stretched) // 2

        # Ensure the window doesn't go out of bounds
        start_idx_edit = max(middle_idx - pre_R_window, 0)
        end_idx_edit = min(middle_idx + post_R_window, len(ecg_data))

        # Extract the centered segment of the stretched beat
        beat_stretched = beat_stretched[start_idx_edit:end_idx_edit]

        # polynomial coefficients for a quadratic polynomial
        a, b, c = 0.1, 0.8, 0

        # normalized range over the stretched beat
        x_stretched_poly = numpy.linspace(0, 1, len(beat))

        polynomial = a * x_stretched_poly**2 + b * x_stretched_poly + c

        beat_augmented = beat * polynomial


        # plot and save the augmented signal
        if closest_annotation != 'N':
            fig, ax = plt.subplots(figsize=(1.66, 1.38), dpi=100)
            ax.plot(beat_stretched, color='black')
            ax.set_title(f'Augmented {closest_annotation}')
            ax.set_xlim(0, len(beat))
            ax.axis('off')
            fig.savefig(f'../../created-images/Edited/{patient_num}_{idx}_new.png', bbox_inches='tight', pad_inches=0)
            print(f'    Patient {patient_num}: original and augmented images of heartbeat {idx} created around index {rpeak_idx}.')
            plt.close(fig)

        if closest_annotation != 'N':
            fig_aug, ax_aug = plt.subplots(figsize=(1.66, 1.38), dpi=100)
            ax_aug.plot(beat_augmented, color='black')
            ax_aug.set_title(f'Polynomial {closest_annotation}')
            ax_aug.set_xlim(0, len(beat))
            ax_aug.axis('off')
            fig_aug.savefig(f'../../created-images/Edited/{patient_num}_{idx}_poly_new.png', bbox_inches='tight', pad_inches=0)
            print(f'    Patient {patient_num}: polynomial augmented image of heartbeat {idx} created around index {rpeak_idx}.')
            plt.close(fig_aug)
