#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "click",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "cmcrameri",
#   "rich",
#   "ase",
#   "pandas",
#   "metatomic",        # Optional, for ideal structures
#   "metatomic-torch",  # Optional, for ideal structures
#   "metatensor-torch", # Optional, for ideal structures
#   "chemiscope",       # Optional, for ideal structures
# ]
# ///

import logging
from pathlib import Path

import ase.io
import click
import cmcrameri.cm as cmc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import sys
from rich.console import Console
from rich.logging import RichHandler
from scipy.interpolate import griddata

import metatomic.torch as mta
import chemiscope

# --- Constants & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
    ],
)
log = logging.getLogger("rich")


# --- Helper Functions (Analysis Logic) ---
def reconstruct_bias_on_grid(hills, grid_x, grid_y):
    """Reconstructs the metadynamics bias potential on a 2D grid from a HILLS file."""
    bias = np.zeros_like(grid_x)
    if len(hills) == 0:
        return bias
    hills_cv1, hills_cv2 = hills[:, 1], hills[:, 2]
    hills_sigma1, hills_sigma2 = hills[:, 3], hills[:, 4]
    hills_height = hills[:, 5]
    for i in range(len(hills)):
        term1 = ((grid_x - hills_cv1[i]) / hills_sigma1[i]) ** 2
        term2 = ((grid_y - hills_cv2[i]) / hills_sigma2[i]) ** 2
        bias += hills_height[i] * np.exp(-0.5 * (term1 + term2))
    return bias


def find_fes_minima(fes_info, nbins=8):
    """Finds local minima on a Free Energy Surface by dividing it into bins."""
    fes = fes_info["fes"]
    rows = fes_info["rows"]
    per = fes_info["per"]
    if rows % nbins != 0:
        raise ValueError(
            f"Error: GRID_BINS ({rows}) must be an integer multiple of MINIMA_NBINS ({nbins})."
        )
    rb = rows // nbins
    if rb < 2:
        raise ValueError("Error: nbins is too high for the grid size, try reducing it.")
    minima_info = []
    for i in range(nbins):
        for j in range(nbins):
            i_start, i_end = i * rb - 1, (i + 1) * rb + 1
            j_start, j_end = j * rb - 1, (j + 1) * rb + 1
            indices_i, indices_j = np.arange(i_start, i_end), np.arange(j_start, j_end)
            if not per[0]:
                indices_i = np.clip(indices_i, 0, rows - 1)
            if not per[1]:
                indices_j = np.clip(indices_j, 0, rows - 1)
            sub_fes = np.take(
                np.take(fes, indices_i, axis=0, mode="wrap" if per[0] else "clip"),
                indices_j,
                axis=1,
                mode="wrap" if per[1] else "clip",
            )

            # If the entire sub-region has no data (all NaN), skip it.
            if np.all(np.isnan(sub_fes)):
                continue

            min_loc_i, min_loc_j = np.unravel_index(
                np.nanargmin(sub_fes), sub_fes.shape
            )
            is_on_border = (
                min_loc_i == 0
                or min_loc_i == len(indices_i) - 1
                or min_loc_j == 0
                or min_loc_j == len(indices_j) - 1
            )
            if not is_on_border:
                global_i, global_j = (
                    indices_i[min_loc_i] % rows,
                    indices_j[min_loc_j] % rows,
                )
                minima_info.append(
                    {
                        "CV1bin": global_i,
                        "CV2bin": global_j,
                        "CV1": fes_info["x"][global_i],
                        "CV2": fes_info["y"][global_j],
                        "free_energy": fes[global_i, global_j],
                    }
                )
    if not minima_info:
        return None
    minima_df = (
        pd.DataFrame(minima_info)
        .drop_duplicates(subset=["CV1bin", "CV2bin"])
        .sort_values(by="free_energy")
        .reset_index(drop=True)
    )
    labels = list(string.ascii_uppercase) + [
        f"{c1}{c2}" for c1 in string.ascii_uppercase for c2 in string.ascii_uppercase
    ]
    minima_df.insert(0, "letter", labels[: len(minima_df)])
    return {"minima": minima_df}


def calculate_tiwary_fes(
    hills_data,
    colvar_data,
    grid_x_mesh,
    grid_y_mesh,
    grid_edges,
    kbt,
    bias_factor,
    n_snapshots=50,
):
    """
    Calculates the FES using the time-dependent reweighting scheme from Tiwary & Parrinello.
    """
    log.info("Calculating time-dependent normalization constant c(t)...")

    # Take snapshots of the bias potential in time
    total_hills = len(hills_data)
    snapshot_indices = np.linspace(0, total_hills, n_snapshots, dtype=int)[1:]

    s1_integrals, s2_integrals = [], []
    for idx in snapshot_indices:
        # Reconstruct bias V(s,t) at this snapshot in time
        bias_at_t = reconstruct_bias_on_grid(hills_data[:idx], grid_x_mesh, grid_y_mesh)

        # Calculate the integrals needed for c(t)
        s1_integrals.append(np.sum(np.exp(bias_at_t / kbt)))
        s2_integrals.append(np.sum(np.exp(bias_at_t / (kbt * bias_factor))))

    c_t = np.array(s1_integrals) / np.array(s2_integrals)
    snapshot_times = hills_data[snapshot_indices - 1, 0]

    # Interpolate c(t) to get a value for every point in the COLVAR trajectory
    log.info("Interpolating c(t) and reweighting trajectory...")
    colvar_times = colvar_data[:, 0]
    c_t_interpolated = np.interp(
        colvar_times, snapshot_times, c_t, left=c_t[0], right=c_t[-1]
    )

    # The instantaneous bias is the 4th column (index 3) of the COLVAR file from METAD
    try:
        instant_bias = colvar_data[:, 3]
    except IndexError:
        log.error(
            "COLVAR file does not have column 4 (index 3) for the instantaneous bias."
        )
        log.error(
            "Cannot perform Tiwary reweighting. Please ensure 'metad.bias' is in COLVAR."
        )
        sys.exit(1)

    weights = np.exp(instant_bias / kbt) / c_t_interpolated

    # Bin the weighted trajectory to get the probability distribution
    prob, _, _ = np.histogram2d(
        colvar_data[:, 1], colvar_data[:, 2], bins=grid_edges, weights=weights
    )

    prob /= np.sum(prob)  # Normalize

    # Convert probability to FES
    fes = np.full_like(prob, np.nan)
    has_data = prob > 1e-15
    fes[has_data] = -kbt * np.log(prob[has_data])
    fes -= np.nanmin(fes)

    # This method doesn't inherently support block averaging for error bars
    # so we return NaN for the error map.
    err = np.full_like(fes, np.nan)

    return fes, err


# --- Helper Functions (Plotting) ---
def setup_plot_aesthetics(ax, title, xlabel, ylabel, fontsizes, facecolor):
    """Applies labels, limits, and other plot aesthetics."""
    ax.set_title(title, fontsize=fontsizes["title"])
    ax.set_xlabel(xlabel, fontsize=fontsizes["label"])
    ax.set_ylabel(ylabel, fontsize=fontsizes["label"])
    ax.tick_params(axis="both", which="major", labelsize=fontsizes["tick"])
    ax.set_facecolor(facecolor)
    ax.minorticks_on()
    ax.grid(False)
    plt.grid(False)


def plot_structure_2d(ax, atoms, title, fontsizes, title_color="black"):
    """
    Helper function to plot a 2D projection of an ASE Atoms object,
    with automatic padding to prevent clipping.
    """
    positions = atoms.get_positions()

    # 1. Center the structure at (0,0) for a consistent view
    positions -= np.mean(positions, axis=0)

    # 2. Calculate plot limits with a 15% margin to prevent clipping
    min_coords = np.min(positions[:, :2], axis=0)
    max_coords = np.max(positions[:, :2], axis=0)
    data_range = max_coords - min_coords
    margin = data_range * 0.15  # 15% padding
    xlims = [min_coords[0] - margin[0], max_coords[0] + margin[0]]
    ylims = [min_coords[1] - margin[1], max_coords[1] + margin[1]]

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=150,
        c="#2a6f97",
        edgecolors="black",
        linewidth=0.5,
    )

    ax.set_title(title, fontsize=fontsizes["label"], color=title_color, weight="bold")

    # Apply the calculated limits
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")


# --- Main Command ---
@click.command()
@click.option(
    "--hills-file",
    default="HILLS",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the HILLS file.",
)
@click.option(
    "--colvar-file",
    default="COLVAR",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the COLVAR file.",
)
@click.option(
    "--trajectory-file",
    default="lj38.lammpstrj",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the trajectory file for structures.",
)
@click.option(
    "--temperature", default=20.5, type=float, help="Simulation temperature in Kelvin."
)
@click.option(
    "--grid-bins",
    default=(100, 100),
    type=(int, int),
    help="Number of bins for the FES grid (CV1, CV2).",
)
@click.option(
    "--reweighting-mode",
    type=click.Choice(["final-bias", "tiwary"], case_sensitive=False),
    default="final-bias",
    help="Algorithm for FES calculation.",
)
@click.option(
    "--bias-factor",
    default=15.0,
    type=float,
    help="Bias factor (gamma), required for 'tiwary' mode.",
)
@click.option(
    "--tiwary-snapshots",
    default=50,
    type=int,
    help="Number of time snapshots for 'tiwary' mode.",
)
@click.option(
    "--n-blocks",
    default=5000,
    type=int,
    help="Number of blocks for error analysis (only for 'final-bias' mode).",
)
@click.option(
    "--find-minima/--no-find-minima",
    default=True,
    help="Enable/disable minima finding.",
)
@click.option(
    "--minima-nbins", default=10, type=int, help="Number of sub-bins for minima search."
)
@click.option(
    "--min-on-reweigh/--min-on-direct-sum",
    default=True,
    help="Use reweighted FES for minima finding (can be noisy, but better).",
)
@click.option(
    "--num-minima-to-plot",
    default=3,
    type=int,
    help="Number of lowest-energy minima to plot.",
)
@click.option(
    "--plot-structures/--no-plot-structures",
    default=True,
    help="Enable/disable plotting of atomic structures.",
)
@click.option(
    "--plot-start-end/--no-plot-start-end",
    default=False,
    help="Plot markers and structures for start/end points.",
)
@click.option(
    "--plot-ideal-structures/--no-plot-ideal-structures",
    default=True,
    help="Plot markers and structures for ideal geometries.",
)
@click.option(
    "--ideal-struct-model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="gen/histo-cv.pt",
    help="Path to model for featurizing ideal structures.",
)
@click.option(
    "--ico-struct-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="data/lj-ico.xyz",
    help="Path to icosahedron geometry file.",
)
@click.option(
    "--oct-struct-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="data/lj-oct.xyz",
    help="Path to octahedron geometry file.",
)
@click.option(
    "-o",
    "--output-file",
    default="final_fes_analysis.png",
    type=click.Path(path_type=Path),
    help="Output image file name.",
)
@click.option(
    "--figsize",
    default=(10.74, 7.0),
    type=(float, float),
    help="Figure size (width, height) in inches.",
)
@click.option(
    "--dpi", default=300, type=int, help="Resolution of the output plot in DPI."
)
@click.option(
    "--fontsize-base", default=12, type=int, help="Base font size for plot text."
)
@click.option(
    "--facecolor", default="floralwhite", help="Background color of the plot axes."
)
def main(**kwargs):
    """
    Analyzes a well-tempered metadynamics simulation, calculates the Free Energy Surface
    with errors, identifies minima, and generates a publication-quality plot.
    """
    # --- 1. Setup & Data Loading ---
    log.info("--- Metadynamics FES Analysis ---")
    KB_KCAL_PER_MOL_K = 0.0019872041
    KBT = KB_KCAL_PER_MOL_K * kwargs["temperature"]
    fontsizes = {
        "title": kwargs["fontsize_base"] + 2,
        "label": kwargs["fontsize_base"],
        "tick": kwargs["fontsize_base"] - 2,
    }

    log.info("Loading simulation data...")
    colvar_data = np.loadtxt(kwargs["colvar_file"])
    hills_data = np.loadtxt(kwargs["hills_file"])

    trajectory = None
    sampling_ratio = 1.0
    if kwargs["plot_structures"]:
        log.info(f"Loading trajectory from [cyan]{kwargs['trajectory_file']}[/cyan]")
        trajectory = ase.io.read(kwargs["trajectory_file"], index=":")
        if len(trajectory) != len(colvar_data):
            log.warning(
                f"Trajectory frames ({len(trajectory)}) do not match COLVAR entries ({len(colvar_data)})."
            )
            sampling_ratio = len(colvar_data) / len(trajectory)
            log.info(
                f"Inferred sampling ratio: 1 frame per {sampling_ratio:.1f} COLVAR points."
            )

    ideal_structures = {}
    ideal_structures_cvs = {}
    featurizer_ch = None
    if kwargs["plot_ideal_structures"] and mta and chemiscope:
        log.info("Loading ideal structures and featurizer...")
        try:
            model_ch = mta.load_atomistic_model(
                kwargs["ideal_struct_model"], extensions_directory="./extensions"
            )
            featurizer_ch = chemiscope.metatomic_featurizer(model_ch)
            ideal_structures["Octahedron"] = ase.io.read(kwargs["oct_struct_file"])
            ideal_structures["Icosahedron"] = ase.io.read(kwargs["ico_struct_file"])
            feats = featurizer_ch(list(ideal_structures.values()), None)
            ideal_structures_cvs["Octahedron"] = feats[0]
            ideal_structures_cvs["Icosahedron"] = feats[1]
        except Exception as e:
            log.error(f"Failed to load ideal structures or model: {e}")
            featurizer_ch = None

    # --- 2. Grid & FES Calculation ---
    time, cv1_traj, cv2_traj = colvar_data[:, 0], colvar_data[:, 1], colvar_data[:, 2]
    min_cv1, max_cv1 = np.min(hills_data[:, 1]), np.max(hills_data[:, 1])
    min_cv2, max_cv2 = np.min(hills_data[:, 2]), np.max(hills_data[:, 2])
    dx, dy = max_cv1 - min_cv1, max_cv2 - min_cv2
    # 1% padding
    xlims = [min_cv1 - 0.01 * dx, max_cv1 + 0.01 * dx]
    ylims = [min_cv2 - 0.01 * dy, max_cv2 + 0.01 * dy]

    grid_x_edges = np.linspace(xlims[0], xlims[1], kwargs["grid_bins"][0] + 1)
    grid_y_edges = np.linspace(ylims[0], ylims[1], kwargs["grid_bins"][1] + 1)
    grid_x_centers = (grid_x_edges[:-1] + grid_x_edges[1:]) / 2
    grid_y_centers = (grid_y_edges[:-1] + grid_y_edges[1:]) / 2
    grid_x_mesh, grid_y_mesh = np.meshgrid(
        grid_x_centers, grid_y_centers, indexing="ij"
    )

    log.info("Reconstructing bias potential for reweighting...")
    if kwargs["reweighting_mode"] == "tiwary":
        fes_reweighted, fes_err = calculate_tiwary_fes(
            hills_data,
            colvar_data,
            grid_x_mesh,
            grid_y_mesh,
            grid_edges=[grid_x_edges, grid_y_edges],
            kbt=KBT,
            bias_factor=kwargs["bias_factor"],
            n_snapshots=kwargs["tiwary_snapshots"],
        )
    else:  # Default to final-bias
        log.info("Using standard final-bias reweighting...")
        final_bias = reconstruct_bias_on_grid(hills_data, grid_x_mesh, grid_y_mesh)
        bias_on_traj = griddata(
            (grid_x_mesh.flatten(), grid_y_mesh.flatten()),
            final_bias.flatten(),
            (cv1_traj, cv2_traj),
            method="cubic",
            fill_value=0,
        )
        weights = np.exp(bias_on_traj / KBT)

        log.info(
            f"Starting block analysis with [magenta]{kwargs['n_blocks']}[/magenta] blocks..."
        )
        block_indices = np.array_split(np.arange(len(time)), kwargs["n_blocks"])
        block_histograms = [
            h / np.sum(h)
            for i in block_indices
            if (
                h := np.histogram2d(
                    cv1_traj[i],
                    cv2_traj[i],
                    bins=[grid_x_edges, grid_y_edges],
                    weights=weights[i],
                )[0]
            ).sum()
            > 1e-10
        ]

        log.info("Averaging block data to calculate final Reweighted FES and error...")
        all_hists = np.stack(block_histograms, axis=-1)
        average_prob, var_prob = (
            np.mean(all_hists, axis=-1),
            np.var(all_hists, axis=-1, ddof=1),
        )
        error_prob = np.sqrt(var_prob / all_hists.shape[-1])
        has_data = average_prob > 1e-15
        fes_reweighted, fes_err = (
            np.full_like(average_prob, np.nan),
            np.full_like(average_prob, np.nan),
        )
        fes_reweighted[has_data], fes_err[has_data] = (
            -KBT * np.log(average_prob[has_data]),
            KBT * error_prob[has_data] / average_prob[has_data],
        )
        fes_reweighted -= np.nanmin(fes_reweighted)

    # --- 3. Minima Finding & Structure Identification ---
    # The direct summation bias is always the same, regardless of reweighting mode
    final_bias_for_minima = -reconstruct_bias_on_grid(
        hills_data, grid_x_mesh, grid_y_mesh
    )
    minima_result = None
    if kwargs["find_minima"]:
        fes_for_minima = (
            fes_reweighted if kwargs["min_on_reweigh"] else final_bias_for_minima
        )
        fes_for_minima -= np.nanmin(fes_for_minima)
        source_name = (
            "Reweighted FES" if kwargs["min_on_reweigh"] else "Direct Summation FES"
        )
        log.info(f"Searching for minima on the {source_name}...")
        try:
            fes_info = {
                "fes": fes_for_minima,
                "rows": kwargs["grid_bins"][0],
                "per": [False, False],
                "x": grid_x_centers,
                "y": grid_y_centers,
            }
            minima_result = find_fes_minima(fes_info, nbins=kwargs["minima_nbins"])
            if minima_result:
                log.info(f"Found {len(minima_result['minima'])} minima.")
        except ValueError as e:
            log.error(f"Could not find minima: {e}")

    structures_to_plot = {}
    if kwargs["plot_structures"]:
        if kwargs["plot_start_end"]:
            structures_to_plot["Start"] = trajectory[0]
            structures_to_plot["End"] = trajectory[-1]
        if minima_result:
            for _, row in minima_result["minima"][
                : kwargs["num_minima_to_plot"]
            ].iterrows():
                distances = np.sqrt(
                    (cv1_traj - row["CV1"]) ** 2 + (cv2_traj - row["CV2"]) ** 2
                )
                idx = int(round(np.argmin(distances) / sampling_ratio))
                structures_to_plot[f"Minimum {row['letter']}"] = trajectory[
                    min(idx, len(trajectory) - 1)
                ]
        if featurizer_ch and kwargs["plot_ideal_structures"]:
            structures_to_plot.update(ideal_structures)

    # --- 4. Plotting ---
    log.info("Generating the final plot...")
    plt.rcParams.update({"font.family": "serif"})
    fig = plt.figure(figsize=kwargs["figsize"])
    ax2 = None
    if kwargs["reweighting_mode"] == "tiwary":
        log.info("Using 2-panel layout for Tiwary mode (FES + Structures).")
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.2], hspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        structure_gs_spec = gs[1, 0]
    else:  # final-bias mode
        log.info("Using 3-panel layout for Final-Bias mode (FES + Error + Structures).")
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1.2], hspace=0.4, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        structure_gs_spec = gs[1, :]

    X, Y = np.meshgrid(grid_x_centers, grid_y_centers, indexing="ij")
    panel_label_props = dict(
        fontsize=fontsizes["title"] + 2, fontweight="bold", va="bottom", ha="right"
    )
    ax1.text(-0.1, 1.02, "(a)", transform=ax1.transAxes, **panel_label_props)
    if ax2:
        ax2.text(-0.1, 1.02, "(b)", transform=ax2.transAxes, **panel_label_props)

    marker_colors = {
        "Start": "lime",
        "End": "magenta",
        "FES Minima": "red",
        "Octahedron": "#ff7b00",
        "Icosahedron": "#B965E1",
    }

    max_energy = np.nanmin([20, np.nanmax(fes_reweighted)])
    levels = np.linspace(0, max_energy, 10)
    cf = ax1.contourf(
        X, Y, fes_reweighted, levels=levels, cmap=cmc.batlow, extend="max"
    )
    ax1.contour(
        X, Y, fes_reweighted, levels=levels, colors="black", linewidths=0.5, alpha=0.5
    )
    fig.colorbar(
        cf, ax=ax1, label="Free Energy (kcal/mol)", format=lambda x, _: f"{x:.3f}"
    ).ax.yaxis.label.set_size(fontsizes["label"])

    if ax2:  # Only plot error map if ax2 exists
        max_err = np.nanmax(fes_err[fes_reweighted < max_energy])
        levels2 = np.linspace(0, 1.0 if np.isnan(max_err) else max_err, 21)
        cf2 = ax2.contourf(X, Y, fes_err, levels=levels2, cmap=cmc.batlow)
        fig.colorbar(
            cf2,
            ax=ax2,
            label="Statistical Error (kcal/mol)",
            format=lambda x, _: f"{x:.4f}",
        ).ax.yaxis.label.set_size(fontsizes["label"])

    axes_to_plot = [ax for ax in [ax1, ax2] if ax is not None]
    for ax in axes_to_plot:
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        if kwargs["plot_start_end"]:
            ax.scatter(
                cv1_traj[0],
                cv1_traj[0],
                s=100,
                c=marker_colors["Start"],
                marker="o",
                edgecolors="black",
                lw=1,
                zorder=5,
            )
            ax.scatter(
                cv1_traj[-1],
                cv2_traj[-1],
                s=100,
                c=marker_colors["End"],
                marker="s",
                edgecolors="black",
                lw=1,
                zorder=5,
            )
        if minima_result:
            df = minima_result["minima"][: kwargs["num_minima_to_plot"]]
            ax.scatter(
                df["CV1"],
                df["CV2"],
                s=30,
                c=marker_colors["FES Minima"],
                marker="*",
                lw=2,
                zorder=5,
            )
        if featurizer_ch and kwargs["plot_ideal_structures"]:
            ax.scatter(
                ideal_structures_cvs["Octahedron"][0],
                ideal_structures_cvs["Octahedron"][1],
                c=marker_colors["Octahedron"],
                marker="v",
                s=60,
                zorder=4,
                edgecolors="white",
                lw=1.5,
            )
            ax.scatter(
                ideal_structures_cvs["Icosahedron"][0],
                ideal_structures_cvs["Icosahedron"][1],
                c=marker_colors["Icosahedron"],
                marker="^",
                s=60,
                zorder=4,
                edgecolors="white",
                lw=1.5,
            )

    if minima_result:
        for _, row in minima_result["minima"][
            : kwargs["num_minima_to_plot"]
        ].iterrows():
            ax1.text(
                row["CV1"] - 1,
                row["CV2"],
                f"{row['letter']}",
                color="#E4F3F4",
                fontsize=fontsizes["label"] - 3,
                fontweight="bold",
                zorder=6,
            )

    setup_plot_aesthetics(
        ax1,
        "Reweighted Free Energy Surface",
        "Collective Variable 1",
        "Collective Variable 2",
        fontsizes,
        kwargs["facecolor"],
    )
    if ax2:
        setup_plot_aesthetics(
            ax2,
            "FES Error Map",
            "Collective Variable 1",
            "",
            fontsizes,
            kwargs["facecolor"],
        )

    if kwargs["plot_structures"] and structures_to_plot:

        def sort_key(item):
            title = item[0]
            if title.startswith("Start"):
                return (0, title)
            if title.startswith("Minimum"):
                return (1, title)
            if title.startswith("End"):
                return (2, title)
            if title.startswith("Ideal"):
                return (3, title)
            return (4, title)

        sorted_structures = sorted(structures_to_plot.items(), key=sort_key)
        gs_structures = gridspec.GridSpecFromSubplotSpec(
            1, len(sorted_structures), subplot_spec=structure_gs_spec, wspace=0.1
        )
        for i, (title, atoms) in enumerate(sorted_structures):
            ax_struct = fig.add_subplot(gs_structures[i])
            if i == 0:
                ax_struct.text(
                    -0.2,
                    1.15,
                    "(c)" if ax2 else "(b)",
                    transform=ax_struct.transAxes,
                    fontsize=fontsizes["title"] + 2,
                    fontweight="bold",
                    va="bottom",
                    ha="right",
                )
            plot_color = "black"
            if title.startswith("Start"):
                plot_color = marker_colors["Start"]
            elif title.startswith("End"):
                plot_color = marker_colors["End"]
            elif title.startswith("Minimum"):
                plot_color = marker_colors["FES Minima"]
            elif title.startswith("Octahedron"):
                plot_color = marker_colors["Octahedron"]
            elif title.startswith("Icosahedron"):
                plot_color = marker_colors["Icosahedron"]
            plot_structure_2d(
                ax_struct, atoms, title, fontsizes, title_color=plot_color
            )

    log.info(f"Saving plot to [green]{kwargs['output_file']}[/green]")
    plt.savefig(kwargs["output_file"], dpi=kwargs["dpi"], bbox_inches="tight")
    log.info("Done.")


if __name__ == "__main__":
    main()
