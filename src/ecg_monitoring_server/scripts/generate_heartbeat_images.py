import os
import matplotlib.pyplot as plt

PRE_R_WINDOW = 128  # Number of samples before the R-peak
POST_R_WINDOW = 128  # Number of samples after the R-peak
DIR = 'data/created_images'

def create_images(ecg_data, rpeaks_indices):

    for idx, rpeak_idx in rpeaks_indices:
        # Define the range for the current beat (centered around the R-peak)
        start_idx = max(rpeak_idx - PRE_R_WINDOW, 0)
        end_idx = min(rpeak_idx + POST_R_WINDOW, len(ecg_data))

        # Extract the heartbeat segment
        beat = ecg_data[start_idx:end_idx]

        # Create the plot
        # 3.31, 3.04 for 256 x 256 sized image
        fig, ax = plt.subplots(figsize=(1.66, 1.67), dpi=100)
        ax.plot(beat, color='black')
        ax.set_xlim(0, len(beat))
        ax.axis('off')

        # if annotation present, creates annotation specific folder 
        # for it if not already existing and adds image to that folder,
        # otherwise just creates and adds to created_images folder

        os.makedirs(DIR, exist_ok=True)

        fig.savefig(
            f'{DIR}/{idx}.png',
            dpi=100,
            bbox_inches=None,
            pad_inches=0
        )

        # Close the plot to free up resources
        plt.close(fig)
