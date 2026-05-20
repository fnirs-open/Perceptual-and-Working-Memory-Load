import os

import matplotlib.pyplot as plt
import mne
import mne_bids
import mne_nirs
import numpy as np

if __name__ == "__main__":
    # Should we plot the raw data for each subject?
    plot_subjects = False

    # Where are our files kept?
    original_folder = "../data/original_nirx_data"

    # Define a task name and a directory where to save the data to
    task = "lettersearch"
    bids_root = "../data/lettersearch_BIDS"
    if not os.path.exists(bids_root):
        os.mkdir(bids_root)

    # Read the files and go through each subject
    nirx_folders = [f.path for f in os.scandir(original_folder) if f.is_dir()]
    for folder in nirx_folders:
        # Get the ID of participant
        subject_id = os.path.split(folder)[-1]
        print(f"Processing folder {folder} for subject {subject_id}")

        # Load the NIRS data for this subject
        raw_intensity = mne.io.read_raw_nirx(folder, verbose=True, preload=False)
        print(raw_intensity.info)
        print(raw_intensity.info["meas_date"])

        # Read aux data from SNIRF File
        snirf_file = [f.path for f in os.scandir(folder) if ".snirf" in f.path]
        if len(snirf_file) > 1:
            print("ERROR: found unexpected amount of snirf files")
        snirf_file = snirf_file[0]
        aux_data = mne_nirs.io.snirf.read_snirf_aux(snirf_file)

        # Give the stims some nice names
        trigger_info = {
            np.str_("1.0"): "practice",
            np.str_("2.0"): "LowPL-LowWM",
            np.str_("3.0"): "HighPL-LowWM",
            np.str_("4.0"): "LowPL-HighWM",
            np.str_("5.0"): "HighPL-HighWM",
        }
        raw_intensity.annotations.rename(trigger_info)
        raw_intensity.annotations.set_durations(5)

        # We should have 52 stims (4 conditions x 10 times + 12 practice)
        # if not, exit out
        if (
            len(raw_intensity.annotations) != 52
            or len(raw_intensity.annotations.count()) != 5
        ):
            print("ERROR: wrong amount of annotations")
            print(raw_intensity.annotations)
            print(raw_intensity.annotations.count())

        # Plot
        if plot_subjects:
            # Hemoglobin timeseries data
            no_channels = len(raw_intensity.ch_names)
            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
            raw_haemo.plot(
                n_channels=10,
                duration=20,
                show_scrollbars=True,
                show_scalebars=True,
                scalings=dict(hbo="1e-4", hbr="1e-4"),
                butterfly=True,
            )
            plt.show()

            # Epoch
            raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.002)
            events, event_dict = mne.events_from_annotations(raw_haemo)
            reject_criteria = dict(hbo=80e-4)
            tmin, tmax = -2.5, 15
            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=tmin,
                tmax=tmax,
                reject=reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=(None, 0),
                preload=True,
                detrend=None,
                verbose=True,
            )
            fig = epochs["LowPL-LowWM"].plot_image(
                combine="mean",
                vmin=-30,
                vmax=30,
                ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])),
            )
            plt.show()
            fig = epochs["HighPL-LowWM"].plot_image(
                combine="mean",
                vmin=-30,
                vmax=30,
                ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])),
            )
            plt.show()
            fig = epochs["LowPL-HighWM"].plot_image(
                combine="mean",
                vmin=-30,
                vmax=30,
                ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])),
            )
            plt.show()
            fig = epochs["HighPL-HighWM"].plot_image(
                combine="mean",
                vmin=-30,
                vmax=30,
                ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])),
            )
            plt.show()

        # First convert to SNIRF and re-read the data from SNIRF
        mne_nirs.io.write_raw_snirf(
            raw_intensity, "TEMP_snirf.snirf", aux_data=aux_data
        )
        snirf_intensity = mne.io.read_raw_snirf("TEMP_snirf.snirf")

        # Write to BIDS format
        bids_path = mne_bids.BIDSPath(subject=subject_id, task=task, root=bids_root)
        mne_bids.write_raw_bids(snirf_intensity, bids_path, overwrite=True)
