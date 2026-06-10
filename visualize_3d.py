import matplotlib.pyplot as plt
import mne
import mne_nirs
import numpy as np
import pandas as pd

# Setup
mne.viz.set_browser_backend("qt")
subjects_dir = str(mne.datasets.sample.data_path()) + "/subjects"
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)
mne.set_config("MNE_NIRS_FOLD_PATH", "data/fOLD-public/Supplementary")

# Fetch anatomical labels
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, accept=True)
labels_combined = mne.read_labels_from_annot(
    "fsaverage", "PALS_B12_Brodmann", "both", subjects_dir=subjects_dir
)

ch_model_df_hbo = pd.read_csv("data/cha_model_df_hbo.csv", index_col=0)
raw_haemo_long = mne.io.read_raw_fif("data/raw.fif")

# Visualize channel-level betas on montage
# https://mne.tools/mne-nirs/dev/generated/mne_nirs.visualisation.plot_nirs_source_detector.html
# One value should be specified per channel in the same order as info.chs.

# Get correct order for channel values
hbo_idx = mne.pick_types(raw_haemo_long.info, fnirs="hbo")
info_hbo = mne.pick_info(raw_haemo_long.info, hbo_idx)

# Plot each condition
conditions = ["HighPL-HighWM", "HighPL-LowWM", "LowPL-HighWM", "LowPL-LowWM"]
for cond in conditions:
    print(f"\nVisualizing condition {cond}\n")

    stats_cond = ch_model_df_hbo.query(f"Condition == '{cond}'").copy()

    # Order stats according to info channel order
    stats_cond["channel_clean"] = stats_cond["ch_name"].str.split(":").str[0]
    stats_cond = stats_cond.set_index("channel_clean")
    stats_cond = stats_cond.reindex(info_hbo["ch_names"])
    stats_cond_coeff = stats_cond["Coef."].values

    # Set up colors based on df values
    values = stats_cond_coeff.copy()
    vmax = np.nanmax(np.abs(values))
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.cm.RdBu_r

    # Zero out non-significant
    mask = stats_cond["Significant"].values
    values[~mask] = 0.0

    # Set up colors for montage
    colors = cmap(norm(values))

    # Set up colors for cortex projection
    clim = dict(kind="value", pos_lims=[0, 3, 5])

    # Plot the projection
    brain = mne_nirs.visualisation.plot_glm_surface_projection(
        raw_haemo_long.copy().pick("hbo"),
        stats_cond,
        colorbar=True,
        clim=clim,
    )

    # Add montage
    brain.add_sensors(
        info_hbo,
        trans="fsaverage",
        fnirs=["channels", "pairs"],
        sensor_colors=colors,
    )

    # Orbitofrontal
    vis_label = [label for label in labels_combined if label.name == "Brodmann.10-lh"][
        0
    ]
    brain.add_label(vis_label, borders=True, color="red")
    vis_label = [label for label in labels_combined if label.name == "Brodmann.10-rh"][
        0
    ]
    brain.add_label(vis_label, borders=True, color="red")

    vis_label = [label for label in labels_combined if label.name == "Brodmann.11-lh"][
        0
    ]
    brain.add_label(vis_label, borders=True, color="green")
    vis_label = [label for label in labels_combined if label.name == "Brodmann.11-rh"][
        0
    ]
    brain.add_label(vis_label, borders=True, color="green")

    # Premotor and SMA
    vis_label = [label for label in labels_combined if label.name == "Brodmann.6-lh"][0]
    brain.add_label(vis_label, borders=True, color="blue")
    vis_label = [label for label in labels_combined if label.name == "Brodmann.6-rh"][0]
    brain.add_label(vis_label, borders=True, color="blue")

    print(
        stats_cond[stats_cond["Significant"] == True][
            ["ch_name", "Coef.", "Significant", "P>|z|"]
        ]
    )

    input("ENTER to continue")
