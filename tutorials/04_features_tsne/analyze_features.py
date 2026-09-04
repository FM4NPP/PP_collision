#!/usr/bin/env python3
"""Analyze m6 features: reconstruct the residual stream, probe it, visualize it.

    INPUT   features.npz from extract_features.py
    OUTPUT  three figures + a printed metric table

    python analyze_features.py features.npz --out figures/

THE ONE THING TO GET RIGHT
--------------------------
`return_z=True` hands you each block's residual-BRANCH output `z_i`, not the running
stream. The stream is what the model actually computes:

    x_i = x_0 + sum_{j<i} z_j

with `x_0` the embedder output, which extract_features.py saves for you. Analyzing `z_i`
directly answers a question nobody asked -- "how well does layer 4's *increment* separate
tracks on its own" -- and gives a misleading answer (the increments look worse with depth,
because a late refinement is small and noisy in isolation).

TWO PROBES, BOTH UNSUPERVISED
-----------------------------
  silhouette    are same-track points compact and well-separated in feature space?
  kNN purity    of a point's 5 nearest neighbours in feature space, how many are on its
                own track? This is closer to what the pretraining objective optimizes
                (predict your neighbours' positions) and to what a tracking head exploits.

Both are geometric probes of the RAW feature space. They are not a verdict on the model:
the real adapter is a query decoder with attention and Hungarian matching, which can read
structure these probes cannot see. Treat a flat probe curve as "Euclidean geometry stops
improving", not "the backbone stops helping" -- downstream ARI does keep improving with
backbone size (m3 0.818 -> m6 0.859 pooled on held-out data).
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from sklearn.decomposition import PCA                 # noqa: E402
from sklearn.manifold import TSNE                     # noqa: E402
from sklearn.metrics import silhouette_score          # noqa: E402
from sklearn.neighbors import NearestNeighbors        # noqa: E402

# The layer-mixing weights the real m6 downstream run learned, for comparison.
# softmax(weighted_avg_weights) from a converged 70k-event track-finding head.
LEARNED_W = np.array([0.0973, 0.1191, 0.0644, 0.0934, 0.0784, 0.0793,
                      0.0882, 0.0752, 0.0755, 0.0782, 0.0778, 0.0732])


def load(path):
    d = np.load(path, allow_pickle=True)
    z = d['features'].astype(np.float32)              # (L, N, D) branch outputs
    x0 = d['x0'].astype(np.float32)                   # (N, D) embedder output
    # x_i = x_0 + sum_{j<i} z_j   -- verified against forward hooks to 1.9e-06
    stream = x0[None] + np.concatenate(
        [np.zeros_like(z[:1]), z.cumsum(0)[:-1]], axis=0)
    final = x0 + z.sum(0)
    return dict(points=d['points'].astype(np.float32), seg=d['seg_target'],
                evt=d['event_id'], z=z, x0=x0, stream=stream, final=final)


def _std(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-6)


def silhouette(X, seg, evt):
    """Mean over events of the per-event silhouette. Noise (track id 0) excluded."""
    vals = []
    for e in np.unique(evt):
        m = (evt == e) & (seg != 0)
        if m.sum() < 10 or len(np.unique(seg[m])) < 2:
            continue
        vals.append(silhouette_score(_std(X[m]), seg[m]))
    return float(np.mean(vals)) if vals else float('nan')


def knn_purity(X, seg, evt, k=5):
    """Fraction of each point's k nearest neighbours that share its true track."""
    vals = []
    for e in np.unique(evt):
        m = (evt == e) & (seg != 0)
        if m.sum() < k + 2:
            continue
        A, lab = _std(X[m]), seg[m]
        idx = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(
            A, return_distance=False)[:, 1:]
        vals.append((lab[idx] == lab[:, None]).mean())
    return float(np.mean(vals)) if vals else float('nan')


def embed2d(X, seed=0):
    """PCA to 50 dims, then t-SNE. Standardize first: features are un-normed."""
    X = _std(X.astype(np.float32))
    if X.shape[1] > 50:
        X = PCA(n_components=min(50, X.shape[0] - 1), random_state=seed).fit_transform(X)
    perp = min(30, max(5, (X.shape[0] - 1) // 4))
    return TSNE(n_components=2, perplexity=perp, init='pca',
                learning_rate='auto', random_state=seed).fit_transform(X)


def scatter(ax, xy, labels, title):
    uniq = np.unique(labels)
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(uniq), 2)))
    for c, u in zip(cmap, uniq):
        m = labels == u
        if u == 0:                                    # noise: grey, small, behind
            ax.scatter(xy[m, 0], xy[m, 1], s=5, c='0.78', alpha=.55, zorder=1)
        else:
            ax.scatter(xy[m, 0], xy[m, 1], s=13, color=c, alpha=.9,
                       edgecolors='none', zorder=2)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('npz')
    ap.add_argument('--out', default='figures')
    ap.add_argument('--k', type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = load(args.npz)
    seg, evt, z, stream = d['seg'], d['evt'], d['z'], d['stream']
    L = z.shape[0]
    ev_ids = np.unique(evt)
    print(f'{len(seg):,} points, {L} layers, dim {z.shape[2]}, {len(ev_ids)} events\n')

    # ------------------------------------------------------------ metrics
    rows = []
    rows.append(('raw (E,eta,phi,r)', silhouette(d['points'], seg, evt),
                 knn_purity(d['points'], seg, evt, args.k)))
    rows.append(('x0 (embedder)', silhouette(d['x0'], seg, evt),
                 knn_purity(d['x0'], seg, evt, args.k)))
    sil_s, pur_s, sil_z, pur_z = [], [], [], []
    for i in range(L):
        sil_s.append(silhouette(stream[i], seg, evt))
        pur_s.append(knn_purity(stream[i], seg, evt, args.k))
        sil_z.append(silhouette(z[i], seg, evt))
        pur_z.append(knn_purity(z[i], seg, evt, args.k))
    rows.append(('final = x0 + sum(z)', silhouette(d['final'], seg, evt),
                 knn_purity(d['final'], seg, evt, args.k)))

    print(f'{"representation":<22}{"silhouette":>12}{"kNN purity":>13}')
    print('-' * 47)
    print(f'{rows[0][0]:<22}{rows[0][1]:>12.4f}{rows[0][2]:>13.4f}')
    print(f'{rows[1][0]:<22}{rows[1][1]:>12.4f}{rows[1][2]:>13.4f}')
    for i in range(L):
        print(f'{"  stream layer " + str(i+1):<22}{sil_s[i]:>12.4f}{pur_s[i]:>13.4f}')
    print(f'{rows[2][0]:<22}{rows[2][1]:>12.4f}{rows[2][2]:>13.4f}')

    print(f'\n{"branch z_i (the WRONG object to probe)":<40}')
    print(f'{"  layer 1":<22}{sil_z[0]:>12.4f}{pur_z[0]:>13.4f}')
    print(f'{"  layer " + str(L):<22}{sil_z[-1]:>12.4f}{pur_z[-1]:>13.4f}')

    # does the head's learned layer mixing track any of this?
    from scipy.stats import pearsonr, spearmanr
    r, p = pearsonr(LEARNED_W, pur_s)
    rs, ps = spearmanr(LEARNED_W, pur_s)
    print(f'\nlearned layer weights vs stream kNN purity: '
          f'pearson r={r:+.3f} (p={p:.2f}), spearman={rs:+.3f} (p={ps:.2f})')
    print(f'  first-6 weight share {LEARNED_W[:6].sum():.3f} vs 0.500 uniform')

    # ------------------------------------------------------------ fig 1
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.6))
    for ax, e in zip(axes.ravel(), ev_ids[:10]):
        m = evt == e
        ntr = len(np.unique(seg[m][seg[m] != 0]))
        scatter(ax, embed2d(d['final'][m]), seg[m],
                f'event {e}  ({m.sum()} pts, {ntr} tracks)')
    fig.suptitle('m6 final residual stream, t-SNE, colored by ground-truth track '
                 '(grey = noise)', fontsize=12)
    plt.tight_layout(); plt.savefig(f'{args.out}/tsne_events.png', dpi=130); plt.close()
    print(f'\nwrote {args.out}/tsne_events.png')

    # ------------------------------------------------------------ fig 2
    big = max(ev_ids, key=lambda e: (evt == e).sum())
    m = evt == big
    show = [0, 3, 7, 11]
    fig, axes = plt.subplots(2, len(show) + 1, figsize=(17, 7))
    scatter(axes[0, 0], embed2d(d['points'][m]), seg[m], 'raw (E,$\\eta$,$\\phi$,r)')
    for ax, li in zip(axes[0, 1:], show):
        scatter(ax, embed2d(stream[li][m]), seg[m], f'stream, entering layer {li+1}')
    scatter(axes[1, 0], embed2d(d['x0'][m]), seg[m], '$x_0$ (embedder output)')
    for ax, li in zip(axes[1, 1:], show):
        scatter(ax, embed2d(z[li][m]), seg[m], f'branch $z_{{{li+1}}}$ only')
    fig.suptitle(f'event {big} ({m.sum()} points) — top: the accumulated stream '
                 f'(right object).  bottom: individual branch outputs (wrong object)',
                 fontsize=12)
    plt.tight_layout(); plt.savefig(f'{args.out}/tsne_layer_sweep.png', dpi=130); plt.close()
    print(f'wrote {args.out}/tsne_layer_sweep.png')

    # ------------------------------------------------------------ fig 3
    xs = np.arange(1, L + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))

    axes[0].plot(xs, pur_s, 'o-', lw=2, color='navy', label='stream $x_i$')
    axes[0].plot(xs, pur_z, 's--', lw=1.6, color='darkorange', label='branch $z_i$')
    axes[0].axhline(rows[1][2], ls=':', c='green', lw=1.8, label='$x_0$ embedder')
    axes[0].axhline(rows[0][2], ls='--', c='crimson', lw=1.6, label='raw coords')
    axes[0].set_xlabel('layer'); axes[0].set_ylabel(f'kNN purity (k={args.k})')
    axes[0].set_title('Neighbourhood purity'); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.3); axes[0].set_xticks(xs)

    axes[1].plot(xs, sil_s, 'o-', lw=2, color='navy', label='stream $x_i$')
    axes[1].plot(xs, sil_z, 's--', lw=1.6, color='darkorange', label='branch $z_i$')
    axes[1].axhline(rows[1][1], ls=':', c='green', lw=1.8, label='$x_0$ embedder')
    axes[1].axhline(rows[0][1], ls='--', c='crimson', lw=1.6, label='raw coords')
    axes[1].set_xlabel('layer'); axes[1].set_ylabel('silhouette')
    axes[1].set_title('Cluster compactness'); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.3); axes[1].set_xticks(xs)

    axes[2].bar(xs, LEARNED_W, color='slateblue', alpha=.85)
    axes[2].axhline(1 / L, ls='--', c='k', lw=1.4, label=f'uniform = {1/L:.4f}')
    axes[2].set_xlabel('layer'); axes[2].set_ylabel('softmax weight')
    axes[2].set_title(f'What the head actually learned\n'
                      f'(r={r:+.2f} with purity, p={p:.2f} — no relation)', fontsize=10)
    axes[2].legend(fontsize=8); axes[2].grid(alpha=.3, axis='y'); axes[2].set_xticks(xs)

    plt.tight_layout(); plt.savefig(f'{args.out}/probes_by_layer.png', dpi=130); plt.close()
    print(f'wrote {args.out}/probes_by_layer.png')

    np.savez(f'{args.out}/metrics.npz', sil_stream=sil_s, pur_stream=pur_s,
             sil_branch=sil_z, pur_branch=pur_z, learned_w=LEARNED_W,
             raw=rows[0][1:], x0=rows[1][1:], final=rows[2][1:])


if __name__ == '__main__':
    main()
