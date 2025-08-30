from biosppy.signals import ecg

def detect_rpeaks(signal, sampling_rate=360):

    # identifies ecg features (including r peaks)
    general_data = ecg.ecg(
        signal=signal,
        sampling_rate=sampling_rate,
        show=False
    )

    return general_data['rpeaks']
