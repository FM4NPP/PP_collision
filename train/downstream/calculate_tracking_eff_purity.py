#!/usr/bin/env python3
"""
Standalone script to calculate tracking efficiency and purity vs pT
from pre-computed CSV predictions.
"""
import argparse
import os

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate tracking efficiency and purity vs pT from CSV predictions."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--eff_threshold", type=float, default=0.5,
                        help="Min pred_assignment_ratio to pass efficiency cut (default: 0.5)")
    parser.add_argument("--pure_threshold", type=float, default=0.5,
                        help="Min purity ratio to pass purity cut (default: 0.5)")
    parser.add_argument("--n_hit_threshold", type=int, default=4,
                        help="Min hits per reconstructed track (default: 4)")
    parser.add_argument("--n_bins", type=int, default=15,
                        help="Number of pT bins (default: 15)")
    parser.add_argument("--pt_min", type=float, default=0.0,
                        help="Lower bound of pT binning range in GeV (default: 0.0)")
    parser.add_argument("--pt_max", type=float, default=3.0,
                        help="Upper bound of pT binning range in GeV (default: 3.0)")
    parser.add_argument("--pt_cut_min", type=float, default=None,
                        help="Min pT cut on truth tracks for efficiency denominator (default: None)")
    parser.add_argument("--pt_cut_max", type=float, default=None,
                        help="Max pT cut on truth tracks for efficiency denominator (default: None)")
    parser.add_argument("--save_csv", action="store_true",
                        help="Save efficiency/purity tables as CSV files")
    parser.add_argument("--output_dir", default=".",
                        help="Directory for output plots and CSVs (default: .)")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip matplotlib plots")
    return parser.parse_args()


def most_popular_in_group(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    """
    For each unique value of `group_col`, finds the most frequent `target_col` value,
    its count, and the ratio of that count to the total rows in that group.
    """
    results = []
    for group_val in df[group_col].unique():
        sub_df = df[df[group_col] == group_val]
        counts = sub_df[target_col].value_counts()
        most_popular = counts.idxmax()
        most_popular_count = counts.max()
        total_count = counts.sum()
        ratio = most_popular_count / total_count
        results.append((group_val, most_popular, most_popular_count, ratio))

    results_df = pd.DataFrame(
        results,
        columns=[
            group_col,
            f"most_popular_{target_col}",
            "most_popular_count",
            "ratio"
        ]
    )
    return results_df


def all_values_ratio_in_group(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    """
    For each unique value of `group_col`, finds all possible values in `target_col`,
    their counts, the total rows in the group, and the ratio of each count to the total.
    """
    grouped = df.groupby([group_col, target_col]).size().reset_index(name='count')
    grouped['total'] = grouped.groupby(group_col)['count'].transform('sum')
    grouped['ratio'] = grouped['count'] / grouped['total']
    grouped = grouped.rename(columns={
        'count': f'{target_col}_count',
        'total': f'{group_col}_total',
        'ratio': f'{target_col}_ratio'
    })
    return grouped


def make_unique_trkid_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per unique seg_target,
    including columns: seg_target, px, py, pz, vtx_x, vtx_y, vtx_z.
    """
    grouped_df = (
        df.groupby('seg_target', as_index=False)
          [['px', 'py', 'pz', 'vtx_x', 'vtx_y', 'vtx_z']]
          .first()
    )
    return grouped_df


def process_method_data(model_data):
    """Merges efficiency + purity data per event batch."""
    all_list = []
    for df in model_data:
        all_predict_df_eff = all_values_ratio_in_group(
            df, group_col='seg_target', target_col='pred_assignment'
        )
        popular_predict_df_pure = most_popular_in_group(
            df, group_col='pred_assignment', target_col='seg_target'
        )
        all_predict_df = all_predict_df_eff.merge(
            popular_predict_df_pure,
            how='left',
            left_on='pred_assignment',
            right_on='pred_assignment',
            suffixes=('_eff', '_pure')
        )
        df_grouped = make_unique_trkid_df(df)
        all_predict_df = all_predict_df.merge(
            df_grouped,
            how='left',
            left_on='seg_target',
            right_on='seg_target'
        )
        all_list.append(all_predict_df)
    return all_list


def calculate_efficiency_new(df_list, eff_threshold, pure_threshold,
                              n_bins=15, pt_min=0.0, pt_max=3.0,
                              pt_cut_min=None, pt_cut_max=None,
                              n_hit_threshold=4):
    """Process a list of DataFrames to calculate efficiency, purity, and their errors vs pT."""
    track_df_list = []
    processed_df_list = []

    for df_temp in df_list:
        df = df_temp.copy()

        # Calculate transverse momentum and pseudorapidity
        df['pT'] = np.sqrt(df['px']**2 + df['py']**2)
        df['eta'] = np.arctanh(df['pz'] / np.sqrt(df['px']**2 + df['py']**2 + df['pz']**2))

        # Determine tracks passing cuts
        df['pass_cut'] = (
            (df['pred_assignment_count'] >= n_hit_threshold) &
            (df['pred_assignment_ratio'] > eff_threshold) &
            (df['ratio'] > pure_threshold) &
            (df['most_popular_seg_target'] == df['seg_target'])
        )

        # Count valid reconstructions per true track
        df['pass_cut_count'] = df.groupby('seg_target')['pass_cut'].transform('sum')
        df['max_pred_label_ratio'] = df.groupby('seg_target')['pred_assignment_ratio'].transform('max')

        # Prepare track-level data (purity)
        track_df = df.drop_duplicates(subset='pred_assignment').copy()
        track_df['enough_hits'] = track_df['pred_assignment_count'] >= n_hit_threshold
        track_df_list.append(track_df)

        # Apply pT cuts on truth tracks for efficiency denominator
        if pt_cut_min is not None:
            df = df[df['pT'] >= pt_cut_min]
        if pt_cut_max is not None:
            df = df[df['pT'] < pt_cut_max]

        # Prepare efficiency data (unique true tracks)
        df = df.drop_duplicates(subset='seg_target', inplace=False)
        processed_df_list.append(df)

    # Combine all data
    combined_df = pd.concat(processed_df_list, ignore_index=True)
    combined_track_df = pd.concat(track_df_list, ignore_index=True)

    bins = np.linspace(pt_min, pt_max, n_bins + 1)
    combined_df['pT_bin'] = pd.cut(combined_df['pT'], bins=bins, include_lowest=True)
    combined_track_df['pT_bin'] = pd.cut(combined_track_df['pT'], bins=bins, include_lowest=True)

    # Calculate efficiency and errors per pT bin
    efficiency_data = []
    for bin_name, bin_group in combined_df.groupby('pT_bin', observed=False):
        valid_tracks = bin_group[(bin_group['seg_target_total'] >= 3)]
        total_tracks = len(valid_tracks)
        efficient_tracks_count = (valid_tracks['pass_cut_count'] >= 1).sum()

        track_hit_efficiency = valid_tracks['max_pred_label_ratio'].mean() if total_tracks > 0 else 0.0
        track_hit_efficiency_error = (
            valid_tracks['max_pred_label_ratio'].std(ddof=1) / np.sqrt(total_tracks)
            if total_tracks > 1 else 0.0
        )

        efficiency = efficient_tracks_count / total_tracks if total_tracks > 0 else 0
        efficiency_error = (
            np.sqrt((efficiency * (1 - efficiency)) / total_tracks) if total_tracks > 0 else 0.0
        )

        non_zero_seeds = valid_tracks[valid_tracks['pass_cut_count'] > 0]['pass_cut_count']
        n_seed_per_track = non_zero_seeds.mean() if not non_zero_seeds.empty else 0

        efficiency_data.append({
            'pT_center': bin_name.mid,
            'bin_width': bin_name.length,
            'efficiency': efficiency,
            'efficiency_error': efficiency_error,
            'n_seed_per_track': n_seed_per_track,
            'n_total_tracks': total_tracks,
            'n_efficient_tracks': efficient_tracks_count,
            'track_hit_efficiency': track_hit_efficiency,
            'track_hit_efficiency_error': track_hit_efficiency_error,
        })

    # Calculate purity and errors per pT bin
    purity_data = []
    for bin_name, bin_group in combined_track_df.groupby('pT_bin', observed=False):
        enough_hits = bin_group['enough_hits']
        subset = bin_group[enough_hits]
        n_purity = len(subset)
        n_pass_cut = subset['pass_cut'].sum()
        fake_rate = 1 - n_pass_cut / n_purity if n_purity > 0 else 0.0

        if n_purity == 0:
            purity = 0.0
            purity_error = 0.0
        else:
            purity = subset['ratio'].mean()
            purity_error = subset['ratio'].std() / np.sqrt(n_purity) if n_purity > 1 else 0.0

        purity_data.append({
            'pT_center': bin_name.mid,
            'purity': purity,
            'purity_error': purity_error,
            'fake_rate': fake_rate,
        })

    return pd.DataFrame(efficiency_data), pd.DataFrame(purity_data)


def plot_efficiency(efficiency_df, output_path=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    pT = efficiency_df['pT_center']
    width = efficiency_df['bin_width']
    eff = efficiency_df['efficiency']
    err = efficiency_df['efficiency_error']

    plt.errorbar(pT, eff,
                 xerr=width / 2,
                 yerr=err,
                 fmt='o',
                 color='#2c7bb6',
                 ecolor='#d7191c',
                 markersize=8,
                 capsize=5,
                 linewidth=2,
                 label='Efficiency')

    plt.xlabel(r'$p_T$ [GeV/c]', fontsize=12)
    plt.ylabel('Efficiency', fontsize=12)
    plt.title('Tracking Efficiency vs Transverse Momentum', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.text(0.95, 0.05, f'Total tracks: {efficiency_df["n_total_tracks"].sum():,}',
             transform=plt.gca().transAxes, ha='right')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Efficiency plot saved to: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_purity(purity_df, output_path=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    pT = purity_df['pT_center']
    purity = purity_df['purity']
    err = purity_df['purity_error']

    plt.errorbar(pT, purity,
                 yerr=err,
                 fmt='s',
                 color='#1a9641',
                 ecolor='#d7191c',
                 markersize=8,
                 capsize=5,
                 linewidth=2,
                 label='Purity')

    plt.xlabel(r'$p_T$ [GeV/c]', fontsize=12)
    plt.ylabel('Purity', fontsize=12)
    plt.title('Tracking Purity vs Transverse Momentum', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Purity plot saved to: {output_path}")
    else:
        plt.show()
    plt.close()


def print_summary(eff_df, purity_df):
    # Integrated efficiency (weighted by number of tracks)
    total_tracks = eff_df['n_total_tracks'].sum()
    total_efficient = eff_df['n_efficient_tracks'].sum()
    integrated_eff = total_efficient / total_tracks if total_tracks > 0 else 0.0
    mean_purity = purity_df['purity'].mean()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total truth tracks (in denominator): {total_tracks:,}")
    print(f"Total efficient tracks:              {total_efficient:,}")
    print(f"Integrated efficiency:               {integrated_eff:.4f}")
    print(f"Mean purity (over pT bins):          {mean_purity:.4f}")

    print("\n--- Efficiency vs pT ---")
    print(eff_df[['pT_center', 'n_total_tracks', 'n_efficient_tracks',
                  'efficiency', 'efficiency_error']].to_string(index=False))

    print("\n--- Purity vs pT ---")
    print(purity_df[['pT_center', 'purity', 'purity_error', 'fake_rate']].to_string(index=False))


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading input: {args.input}")
    df = pd.read_csv(args.input)
    dfs_by_batch = [batch_df for _, batch_df in df.groupby('batch_idx')]
    print(f"  Number of events (batches): {len(dfs_by_batch)}")

    print("Processing method data...")
    processed_list = process_method_data(dfs_by_batch)

    print("Calculating efficiency and purity...")
    eff_df, purity_df = calculate_efficiency_new(
        processed_list,
        eff_threshold=args.eff_threshold,
        pure_threshold=args.pure_threshold,
        n_bins=args.n_bins,
        pt_min=args.pt_min,
        pt_max=args.pt_max,
        pt_cut_min=args.pt_cut_min,
        pt_cut_max=args.pt_cut_max,
        n_hit_threshold=args.n_hit_threshold,
    )

    print_summary(eff_df, purity_df)

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]

    if args.save_csv:
        eff_csv = os.path.join(args.output_dir, f"{stem}_efficiency.csv")
        purity_csv = os.path.join(args.output_dir, f"{stem}_purity.csv")
        eff_df.to_csv(eff_csv, index=False)
        purity_df.to_csv(purity_csv, index=False)
        print(f"\nCSVs saved:\n  {eff_csv}\n  {purity_csv}")

    if not args.no_plot:
        eff_plot = os.path.join(args.output_dir, f"{stem}_efficiency.png")
        purity_plot = os.path.join(args.output_dir, f"{stem}_purity.png")
        plot_efficiency(eff_df, output_path=eff_plot)
        plot_purity(purity_df, output_path=purity_plot)
