import os
from itertools import compress

import mne
import numpy as np
from mne.preprocessing.nirs import optical_density
from mne_bids import BIDSPath, read_raw_bids

# BIDS dataset and task
task = "lettersearch"
bids_root = "data/lettersearch_BIDS"
datatype = "nirs"
bids_folder = BIDSPath(root=bids_root, datatype=datatype)
nirs_files = bids_folder.match()

# Get all subjects in a sorted list
all_subjects = [file.subject for file in nirs_files]
all_subjects = sorted(list(set(all_subjects)))
print("Subjects")
print(all_subjects)

# Go through data for each subject
long_sci_per_participant = []
bad_count_per_participant = []
for subject in all_subjects:
    # Load
    bids_path = BIDSPath(
        subject=subject,
        task=task,
        datatype=datatype,
        root=bids_root,
    )
    print("Using BIDS file path..")
    print(bids_path)
    if not os.path.exists(bids_path):
        print("No SNIRF file, skip: " + str(bids_path))
        continue
    raw_intensity = read_raw_bids(bids_path=bids_path, verbose=False)

    # Which channels are short channels? Have 16 (8 for 850, 8 for 760)
    short_idx = mne.preprocessing.nirs.short_channels(raw_intensity.info)
    short_idx = np.where(short_idx)[0]
    long_ch_names = np.array(raw_intensity.ch_names)
    np.delete(long_ch_names, short_idx)

    # Calculate overall SCI over entire length of signal
    od = optical_density(raw_intensity)
    overall_sci = mne.preprocessing.nirs.scalp_coupling_index(od)
    overall_sci = np.array(overall_sci)
    overall_sci_long = np.delete(overall_sci, short_idx, axis=0)

    bad_channels = list(compress(long_ch_names, overall_sci < 0.8))
    bad_channels = np.unique(
        [
            channel.replace(" 760", "").replace(" 850", "")
            for channel in np.unique(bad_channels)
        ]
    ).tolist()

    # SCI of long channels for this subject
    long_sci_per_participant.append(overall_sci_long)

    # Number of bad channels for this subject
    n_bad = len(bad_channels)
    bad_count_per_participant.append(n_bad)


def median_iqr(data):
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return median, q1, q3, iqr


# Long channels per participant
median_long, q1_long, q3_long, iqr_long = median_iqr(long_sci_per_participant)

# Bad channels per participant
median_bad, q1_bad, q3_bad, iqr_bad = median_iqr(bad_count_per_participant)

print(
    f"Long channels per participant: median={median_long}, IQR={iqr_long} (Q1={q1_long}, Q3={q3_long})"
)
print(
    f"Bad channels per participant: median={median_bad}, IQR={iqr_bad} (Q1={q1_bad}, Q3={q3_bad})"
)
