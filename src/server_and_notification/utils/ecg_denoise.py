"""
This module takes in the ECG data and returns the signal in a 
denoised form, by decomposing the signal into ten sub signals
and removing the low signal and the two highest signals. 
The amiunt of signals being removed can be changed to 
create greater effects of denoising. Finally it inverses the 
wavelet transform which returns it to a signle signal.
"""

import pywt
import numpy


def denoise_signal(ecg_data):
    """denoises the given signal"""

    # decompose signal into wavelt coefficients
    coefficients = pywt.wavedec(ecg_data, 'db5', level=9)

    low_cutoff = 2
    high_cutoff = 7

    # zero out coeffocoents from 0 to low cutoff
    for num in range(0, low_cutoff):
        coefficients[num] = numpy.multiply(coefficients[num],[0.0])

    # zero out coefficients from high cutoff to end
    for num in range(high_cutoff, len(coefficients)):
        coefficients[num] = numpy.multiply(coefficients[num], 0)

    # inverse wavelet transform to reconstruct the signal, now denoised
    denoised_ecg = pywt.waverec(coefficients, 'db5')

    return denoised_ecg
