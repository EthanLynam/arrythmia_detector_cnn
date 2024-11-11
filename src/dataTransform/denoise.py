from pywt import wavedec
import pywt
import numpy 


def denoise_signal(ECG_data):
   
    coefficients = wavedec(ECG_data, 'bior4.4', level=10)  # decompose signal into wavelt coefficients

    low_cutoff = 1
    high_cutoff = 7

    # zero out coeffocoents from 0 to low cutoff 
    for num in range(0, low_cutoff):
        coefficients[num] = numpy.multiply(coefficients[num],[0.0])
    
    # zero out coefficients from high cutoff to end
    for num in range(high_cutoff, len(coefficients)):
        coefficients[num] = numpy.multiply(coefficients[num], 0)

    denoised_ECG = pywt.waverec(coefficients, 'bior4.4')  # inverse wavelet transform to reconstruct the signal, now denoised
    return denoised_ECG
