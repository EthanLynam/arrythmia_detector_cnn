from collections import deque
from threading import Thread
import os
import gc
import asyncio
import websockets
import numpy as np

from scripts.ecg_denoise import denoise_signal
from scripts.ecg_baseline_wander import remove_baseline_wander
from scripts.detect_rpeaks import detect_rpeaks
from scripts.generate_heartbeat_images import create_images
from scripts.heartbeat_classification import classify_heartbeat

# Shared queue to store incoming data
ecg_queue = deque(maxlen=1625)  # scipy signal processing requires minimum 1624 data points.
counter = 0
processing = False  # Flag to indicate if processing is happening

def process_data():

    global ecg_queue, counter, processing
    print("Processing data...")

    processing = True
    ecg_data = np.array(ecg_queue) # convert the queue to array

    ecg_data2 = denoise_signal(ecg_data) # denoise the signal
    ecg_data3 = remove_baseline_wander(ecg_data2) # remove baseline wander
    r_peaks_indices = detect_rpeaks(ecg_data3) # detect R-peaks

    # generate images around detected peaks
    create_images(ecg_data, r_peaks_indices)

    for filename in os.listdir('data/created_images'):

        image_path = os.path.join('data/created_images', filename)
        classify_heartbeat(image_path)
        os.remove(image_path)

    # reset counter and clear queue
    counter = 0
    ecg_queue.clear()

    processing = False  # Reset flag after processing is done

    # force garbage collection to prevent error from matplotlib being
    # used in secondary thread
    gc.collect()

    print("Review results. Data queue cleared and receiving next batch.")


async def handle_connection(websocket):

    global counter, ecg_queue, processing
    print("Client connected.")
    print("Receiving data...")

    try:
        async for message in websocket:

            ecg_value = int(message) # convert message to int (arduino sending as string)
            ecg_queue.append(ecg_value)  # add data to the queue
            counter += 1 # increase counter

            # Although plenty of time is given to process data,
            # should the thread still be busy the data will just
            # continue to be recieved at the cost of some loss
            if counter >= 1625 and not processing:
                # start thread to process the data, as processing on same
                # will create data loss as arduino continuously sends data
                print("Queue full!")
                Thread(target=process_data).start()
                processing = True

    except websockets.exceptions.ConnectionClosedError:
        print() # This error follows straight after client connection,
        # However the connection remains, hence it is caught and ignored.


async def main():

    # start websocket server on 0.0.0.0 and port 9000
    server = await websockets.serve(handle_connection, "0.0.0.0", 9000)
    print("Websocket server started on 0.0.0.0 & Port 9000...")

    # keep server running
    await server.wait_closed()

# run the server
asyncio.run(main())
