from pywt import wavedec
import pywt
import numpy 


def denoise_signal(ECG_data):
    """
    in this
    work, WT is utilized to analyze the component of
    specific frequency sub-bands and to further remove the
    noise.
    In the first place, the Daubechies-5 (db5) mother
    wavelet is utilized to decompose the ECG signals into
    nine high frequency sub-bands and one low frequency
    sub-band. After that, we remove the top three highfrequency sub-bands and one low frequency sub-band,
    then the remaining detailed coefficient sub-bands of the
    fourth, the fifth, the sixth, the seventh, the eighth, and
    the ninth are adopted to reconstruct filtered signal by
    wavelet inverse transform
    """
   
    coefficients = pywt.wavedec(ECG_data, 'db5', level=10)  # decompose signal into wavelt coefficients

    low_cutoff = 1
    high_cutoff = 8

    # zero out coeffocoents from 0 to low cutoff 
    for num in range(0, low_cutoff):
        coefficients[num] = numpy.multiply(coefficients[num],[0.0])
    
    # zero out coefficients from high cutoff to end
    for num in range(high_cutoff, len(coefficients)):
        coefficients[num] = numpy.multiply(coefficients[num], 0)

    denoised_ECG = pywt.waverec(coefficients, 'db5')  # inverse wavelet transform to reconstruct the signal, now denoised

    return denoised_ECG
