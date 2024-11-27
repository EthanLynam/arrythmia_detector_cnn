"""
If inputted beat is not normal, this function will
create 4 new images of the given beat but in a slightly altered
form. It does this by stretching the beat out and multiplying
both the new stretched beat and the original beat by an 
individual positive and negative polynomial. This slightly
alters the beat in 4 different ways, technically creating 4
brand new beats.
"""

import os
import numpy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def beat_augment(beat_data, beat_label, idx, patient_num):
    """creates 4 slightly altered versions of beat_data"""

    if beat_label == 'N':
        return

    # take the original beat and stretch it by 1.25
    x_original = numpy.arange(len(beat_data))
    x_stretched = numpy.linspace(0, len(beat_data) - 1, int(len(beat_data) * 1.25))
    interpolator = interp1d(x_original, beat_data, kind='linear')
    beat_stretched = interpolator(x_stretched)

    # recentre the stretched beats, as stretching moves them to the right
    # on the x axis
    middle_idx = len(beat_stretched) // 2
    start_idx_edit = max(middle_idx - 128, 0)
    end_idx_edit = min(middle_idx + 128, len(beat_stretched))
    beat_stretched = beat_stretched[start_idx_edit:end_idx_edit]

    # polynomials for original beat
    x_original_poly = numpy.linspace(0, 1, len(beat_data))
    poly_positive = 0.3 * x_original_poly**2 + 0.7 * x_original_poly + 0.2
    poly_negative = -0.2 * x_original_poly**2 - 0.5 * x_original_poly + 0.8

    # polynomials for stretched beat
    x_stretched_poly = numpy.linspace(0, 1, len(beat_stretched))
    poly_positive_stretched = 0.4 * x_stretched_poly**2 + 0.9 * x_stretched_poly + 0.3
    poly_negative_stretched = -0.3 * x_stretched_poly**2 - 0.6 * x_stretched_poly + 0.7

    # multiply normal beat by positive and negative polynomials, creating
    # two new images
    new_positive_beat = beat_data * poly_positive
    new_negative_beat = beat_data * poly_negative

    # multiply stretched beat by positive and negative polynomials, creating
    # two other new images
    new_positive_beat_stretched = beat_stretched * poly_positive_stretched
    new_negative_beat_stretched = beat_stretched * poly_negative_stretched

    # Create directories
    direct = f'../../created-images/Edited/Patient_{patient_num}/Beat_{idx}'
    os.makedirs(direct, exist_ok=True)

    # Add the normal beat to the list of images to save
    pairs = [
        (beat_data, "normal"),  # Normal beat
        (new_positive_beat, "original_p"),
        (new_negative_beat, "original_n"),
        (new_positive_beat_stretched, "stretched_p"),
        (new_negative_beat_stretched, "stretched_n"),
    ]

    # Save all images
    for data, name in pairs:
        fig, ax = plt.subplots(figsize=(1.66, 1.38), dpi=100)
        ax.plot(data, color='black')
        ax.set_title(name)
        ax.set_xlim(0, len(data))
        ax.axis('off')

        fig.savefig(
            f'../../created-images/Edited/Patient_{patient_num}/Beat_{idx}/{name}.png',
            bbox_inches='tight',
            pad_inches=0
            )

        plt.close(fig)

    print("      Augmented beats created.")
