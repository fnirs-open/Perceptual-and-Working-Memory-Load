import os
import matplotlib.pyplot as plt
import mne
import mne_bids
import mne_nirs
import numpy as np
import pandas as pd
import seaborn as sns

# https://mne.tools/mne-nirs/dev/generated/mne_nirs.visualisation.plot_nirs_source_detector.html
# One value should be specified per channel in the same order as info.chs.

# Fetch anatomical labels
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, accept=True)
labels_combined = mne.read_labels_from_annot('fsaverage', 'PALS_B12_Brodmann', 'both', subjects_dir=subjects_dir)

# Get correct order for channel values
hbo_idx = mne.pick_types(raw_haemo_long.info, fnirs="hbo")
info_hbo = mne.pick_info(raw_haemo_long.info, hbo_idx)

# Plot each condition
conditions = ['HighPL-HighWM', 'HighPL-LowWM', 'LowPL-HighWM', 'LowPL-LowWM']
#conditions = ['HighPL-HighWM']
for cond in conditions:

    print(f"\nVisualizing condition {cond}\n")

    stats_cond = (
        ch_model_df_hbo
        .query(f"Condition == '{cond}'")
        .copy()
    )

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
    clim = dict(kind="value", lims=[-vmax, 0, vmax])

    # Plot the projection
    brain = mne_nirs.visualisation.plot_glm_surface_projection(
        raw_haemo_long.copy().pick("hbo"),
        stats_cond,
        colorbar=True,
        #clim=clim,
    )

    # Add montage
    brain.add_sensors(
        info_hbo,
        trans="fsaverage",
        fnirs=["channels", "pairs"],
        sensor_colors = colors,
    )

    # Dorsolateral prefrontal
    vis_label = [label for label in labels_combined if label.name == 'Brodmann.9-lh'][0]
    brain.add_label(vis_label, borders=True, color='purple')
    vis_label = [label for label in labels_combined if label.name == 'Brodmann.9-rh'][0]
    brain.add_label(vis_label, borders=True, color='purple')
    vis_label = [label for label in labels_combined if label.name == 'Brodmann.46-lh'][0]
    brain.add_label(vis_label, borders=True, color='purple')
    vis_label = [label for label in labels_combined if label.name == 'Brodmann.46-rh'][0]
    brain.add_label(vis_label, borders=True, color='purple')

    # Frontopolar
    vis_label = [label for label in labels_combined if label.name == 'Brodmann.10-lh'][0]
    brain.add_label(vis_label, borders=True, color='red')
    vis_label = [label for label in labels_combined if label.name == 'Brodmann.10-rh'][0]
    brain.add_label(vis_label, borders=True, color='red')

    # FEF
    vis_label = [label for label in labels_combined if label.name == 'Brodmann.8-lh'][0]
    brain.add_label(vis_label, borders=True, color='cyan')
    vis_label = [label for label in labels_combined if label.name == 'Brodmann.8-rh'][0]
    brain.add_label(vis_label, borders=True, color='cyan')

    # IFG
    vis_label = [label for label in labels_combined if label.name == 'Brodmann.44-lh'][0]
    brain.add_label(vis_label, borders=True, color='green')
    vis_label = [label for label in labels_combined if label.name == 'Brodmann.44-rh'][0]
    brain.add_label(vis_label, borders=True, color='green')

    # Premotor and SMA
    vis_label = [label for label in labels_combined if label.name == 'Brodmann.6-lh'][0]
    brain.add_label(vis_label, borders=True, color='blue')
    vis_label = [label for label in labels_combined if label.name == 'Brodmann.6-rh'][0]
    brain.add_label(vis_label, borders=True, color='blue')

    print(stats_cond[stats_cond["Significant"] == True][["ch_name", "Coef.", "Significant", "P>|z|"]])
