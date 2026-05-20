import os
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import mne
import mne_bids
import mne_nirs
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d


def update(val):
    """
    Slider callback: shift all stimulus blocks by the selected lag.
    """
    new_lag = lag_slider.val
    for rect, onset, dur in zip(stim_patches, onsets, durations):
        rect.set_x(onset + new_lag)  # shift in data coords
        rect.set_width(
            dur
        )  # keep duration (you can make this another slider if needed)
    fig.canvas.draw_idle()


def make_span(ax, x0, duration, **kwargs):
    """
    Create a rectangle that spans the full height of the axes (y from 0 to 1 in Axes coords)
    while x is in data coordinates. This mimics axvspan but is easier to update later.
    """
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    rect = Rectangle(
        (x0, 0),
        duration,
        1,  # (x, y), width, height
        transform=trans,
        edgecolor="none",
        zorder=-100,
        **kwargs,
    )
    ax.add_patch(rect)
    return rect


# Go through PsychoPy data files and convert
# them into a format MNE can read as annotations
if __name__ == "__main__":
    # Where are our files kept?
    annot_folder = "../data/psychopy_triggers"
    annot_files = [f.path for f in os.scandir(annot_folder) if f.name.endswith(".csv")]
    snirf_folder = "../data/original_nirx_data"

    # We have some rows which contains data for pauses and other metadata we don't need
    metadata_rows = [1, 2, 29, 30, 44, 58]

    # Define the stim order
    # Which order are the trials in?
    # For the purposes of creating a .tri file,
    # keep condition names as integers.
    # Give them better names when creating BIDS
    # 1 = practice
    # 2 = LowPL-LowWM
    # 3 = HighPL-LowWM
    # 4 = LowPL-HighWM
    # 5 = HighPL-HighWM
    trial_order = {
        1: [1, 2, 3, 4, 5, 6, 27, 28, 29, 40, 41, 42],
        2: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        3: [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
        4: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        5: [43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
    }

    # For labelling the dataframe
    index_to_label = {
        idx - 1: label for label, lst in trial_order.items() for idx in lst
    }

    for file in annot_files:
        subject = os.path.split(file)[-1]
        subject = subject.replace(".csv", "")
        print(f"Converting events {file} for {subject}")

        annotations_data = pd.read_csv(file, sep=",", header=0, skiprows=metadata_rows)
        annotations_data = annotations_data[
            [
                "participant",
                "date",
                "expStart",
                "trial_marker",
                "trial.started",
                "fixation.started",
                "presentSet.started",
                "resp.started",
                "feedback.started",
            ]
        ]
        annotations_data["trial_type"] = annotations_data.index.map(index_to_label)

        # Read SNIRF file for SNIRF timing
        subject_snirf_folder = f"{snirf_folder}/{subject}"
        snirf_file = [
            f.path
            for f in os.scandir(subject_snirf_folder)
            if f.name.endswith(".snirf")
        ][0]
        raw_intensity = mne.io.read_raw_snirf(snirf_file)

        # Check if we have already created a trigger file for this subject
        trigger_file = [
            f.path for f in os.scandir(subject_snirf_folder) if f.name.endswith(".tri")
        ]
        if trigger_file:
            print("Trigger file already exists, skipping")
            continue

        # What is our sampling frequency?
        snirf_freq = raw_intensity.info["sfreq"]

        # What times do we have?
        snirf_start_time = raw_intensity.info["meas_date"]
        psychopy_start_time = annotations_data["expStart"][0]
        datetime_format_psychopy = "%Y-%m-%d %Hh%M.%S.%f %z"
        psychopy_start_time = datetime.strptime(
            psychopy_start_time, datetime_format_psychopy
        )
        print(f"SNIRF started recording at: {snirf_start_time}")
        print(f"Experiment started global glock at: {psychopy_start_time}")

        # Get the accelerometer aux.
        aux_df = mne_nirs.io.snirf.read_snirf_aux_data(snirf_file, raw_intensity)
        aux_acc = aux_df[["ACCEL_X_1", "ACCEL_Y_1", "ACCEL_Z_1"]].to_numpy()

        # Get the raw SNIRF data as a numpy array
        snirf_data = raw_intensity.get_data()

        # Create time axes
        t_snirf = raw_intensity.times

        # Plot accelerometer and hemo data in two subplots
        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        for ch in snirf_data[0:20]:
            axs[0].plot(t_snirf, ch, alpha=0.3)
        axs[0].set_title("Butterfly Plot – Haemoglobin Channels")
        axs[0].set_ylabel("Amplitude")
        axs[1].plot(t_snirf, aux_acc[:, 0], label="Accelerometer X", color="orange")
        axs[1].set_title("Accelerometer")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Acceleration (g)")
        axs[1].legend()

        # Leave room at the bottom for the slider (avoid overlapping tight_layout)
        plt.subplots_adjust(bottom=0.22)

        # Create stimulus overlay blocks
        stim_patches = []
        onsets = []
        durations = []
        for _, row in annotations_data.iterrows():
            onset = float(row["trial.started"])
            duration = 10.0

            rect_top = make_span(
                axs[0], onset, duration, facecolor="lightseagreen", alpha=0.2
            )
            rect_bottom = make_span(
                axs[1], onset, duration, facecolor="lightseagreen", alpha=0.2
            )

            # append both to the same list
            stim_patches.append(rect_top)
            stim_patches.append(rect_bottom)

            # duplicate onset/duration so indexing matches
            onsets.append(onset)
            durations.append(duration)
            onsets.append(onset)
            durations.append(duration)

        # Slider to adjust onset lag
        slider_min = -200.0
        slider_max = 50.0
        axcolor = "lightgoldenrodyellow"

        # Place the slider nicely below the plots
        slider_axis = fig.add_axes([0.15, 0.08, 0.7, 0.04], facecolor=axcolor)
        lag_slider = Slider(
            slider_axis, "Lag (s)", slider_min, slider_max, valinit=0.0, valstep=0.1
        )
        lag_slider.on_changed(update)
        plt.show()

        # Save adjusted onset and lag between signals
        lag = lag_slider.val
        annotations_data["trial.started.adjusted"] = (
            annotations_data["trial.started"] + lag
        )
        print(f"The time diff between systems should be adjusted by {lag} seconds.")
        print(annotations_data[["trial.started", "trial.started.adjusted"]])

        # Create a trigger file from adjusted data
        annotations_data["trial.started.sample"] = np.floor(
            annotations_data["trial.started.adjusted"] * snirf_freq
        ).astype(int)
        annotations_data["trial.started.global"] = (
            psychopy_start_time
            + pd.to_timedelta(annotations_data["trial.started"], unit="s")
        )
        trigger_data = annotations_data[
            ["trial.started.global", "trial.started.sample", "trial_type"]
        ]

        # Format according to expected trigger structure
        trigger_data["trial.started.global"] = (
            trigger_data["trial.started.global"]
            .dt.tz_localize(None)
            .dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        )
        print(trigger_data)

        # Save trigger file
        tri_file_name = f"{subject_snirf_folder}/{subject}.tri"
        print(f"Saving trigger file to: {tri_file_name}")
        trigger_data.to_csv(tri_file_name, sep=";", index=False, header=False)

    print("All done")
