# VertexHead: the mathematics

How [`train/downstream/vertexing`](../train/downstream/vertexing) turns the track finder's output into primary and
secondary vertices, equation by equation: ACTS's adaptive multi-vertex finder and fitter (with soft cuts), sPHENIX's
secondary-vertex pair selection, and the design notes (§9). Code references are
to the current `vertexhead-extension` branch.

```
hits + mask_probs ──► hard assignment ──► helix hit fit (Taubin circle, z vs arc length) + covariance
                                                   │
                     use_acts ┌────────────────────┴───────────────┐ no ACTS
                              ▼                                    ▼
           ACTS Kalman refit on hit planes                 perigee from the circle
           ACTS backward propagation (material)                    │
                              └────────────────┬───────────────────┘
                                               ▼
                     perigee (d0, z0, φ, θ, q/p) + σ(d0), σ(z0) ──► usable, weight
                                               ▼
     ACTS AMVF: density seed ─► preselection ─► adaptive multi-vertex fit (Kalman, linearised helices) ─► keep? ─► repeat
                                               ▼
              SV candidates ─► circle-intersection pair seeds ─► SV fits ─► PV refit with the SVs (shared tracks)
```

## 1. Conventions

| quantity | unit | note |
|---|---|---|
| positions (our side) | cm | the hits' unit |
| positions (ACTS) | mm | `MM_PER_CM = 10` at the boundary |
| momentum | GeV | $p_T = c\,|B_z|\,R$, $c = 0.299792458$ GeV/(T·m) |
| perigee | $(d_0, z_0, \phi, \theta, q/p, t)$ | ACTS bound parameters, reference = beam line $(b_x, b_y)$ |

Circle sign: $R_s = \operatorname{sign}(qB_z)\,R$; travel is clockwise (seen from $+z$) for $R_s>0$, anticlockwise for
$R_s<0$. The perigee point is $\mathbf{p}_\perp = \mathbf{b} + d_0(-\sin\phi, \cos\phi)$ (ACTS's sign of $d_0$).

## 2. Hits and assignment ([tracks.py:132](../train/downstream/vertexing/tracks.py#L132))

The dataset's normalisation is inverted in float64: $E = E_n\sigma_E + \mu_E$; $\eta, \phi, r$ from their min–max
ranges; then $x = r\cos\phi$, $y = r\sin\phi$, $z = r\sinh\eta$.

Each real hit goes to $\arg\max_t$ `mask_probs` if that maximum is $>0$ (hard assignment; soft probabilities are not
used by the geometry, only `class_probs` enters as a weight, §6).

With a field map, a track's field is the mean of $B_z$ over its hits (trilinear interpolation, clamped to the
grid, [tracks.py:295](../train/downstream/vertexing/tracks.py#L295)).

## 3. Helix hit fit ([tracks.py:151](../train/downstream/vertexing/tracks.py#L151))

### 3.1 Circle (Taubin, Newton form; [tracks.py:72](../train/downstream/vertexing/tracks.py#L72))

With hits centred on the track's centroid, $X_i, Y_i$, $Z_i = X_i^2 + Y_i^2$, and moments
$M_{ab} = \langle ab \rangle$:

$$
M_z = M_{xx} + M_{yy},\quad
\mathrm{Cov}_{xy} = M_{xx}M_{yy} - M_{xy}^2,\quad
\mathrm{Var}_z = M_{zz} - M_z^2
$$

$$
P(x) = A_3x^3 + A_2x^2 + A_1x + A_0,\qquad
\begin{aligned}
A_3 &= 4M_z, & A_2 &= -3M_z^2 - M_{zz},\\
A_1 &= \mathrm{Var}_z M_z + 4\,\mathrm{Cov}_{xy} M_z - M_{xz}^2 - M_{yz}^2, &
A_0 &= M_{xz}(M_{xz}M_{yy} - M_{yz}M_{xy}) + M_{yz}(M_{yz}M_{xx} - M_{xz}M_{xy}) - \mathrm{Var}_z\mathrm{Cov}_{xy}
\end{aligned}
$$

The smallest root $x^*$ by Newton from $x=0$ (≤ 20 steps, kept only while $|P|$ decreases), then with
$D = x^{*2} - x^*M_z + \mathrm{Cov}_{xy}$:

$$
X_c = \frac{M_{xz}(M_{yy}-x^*) - M_{yz}M_{xy}}{2D},\quad
Y_c = \frac{M_{yz}(M_{xx}-x^*) - M_{xz}M_{xy}}{2D},\quad
R = \sqrt{X_c^2 + Y_c^2 + M_z}
$$

and the centre is $(X_c, Y_c)$ + centroid. All hits have equal weight. Fewer than `min_hits` (3) → NaN.

### 3.2 Direction of travel and charge

From the innermost hit $\mathbf{h}_\text{in}$ to the outermost $\mathbf{h}_\text{out}$ (by $r$):
anticlockwise iff $[(\mathbf{h}_\text{in}-\mathbf{c}) \times (\mathbf{h}_\text{out}-\mathbf{c})]_z > 0$, giving
$\operatorname{sign} R_s$ and $q = \operatorname{sign}(R_s)\operatorname{sign}(B_z)$.

### 3.3 Arc length ([common.py:9](../train/downstream/vertexing/common.py#L9))

Measured from the circle's closest approach to the beam line, positive along travel:

$$
s = -R_s\,\operatorname{wrap}_{(-\pi,\pi]}\!\left(\alpha - \alpha_b\right),\qquad
\alpha = \operatorname{atan2}(y - y_c, x - x_c),\quad
\alpha_b = \operatorname{atan2}(b_y - y_c, b_x - x_c)
$$

The wrap limits $|s| \le \pi|R|$: a track turning more than half a circle from its perigee (a looper,
$p_T \lesssim 0.16$ GeV at 1.4 T for the TPC's 75 cm) cannot be described.

### 3.4 z line ([tracks.py:47](../train/downstream/vertexing/tracks.py#L47))

A helix is linear in $s$: $z = s\cot\theta + z_0$. Fitted per track by **Deming regression with equal variances**
(orthogonal distance), with $S_{uu}, S_{vv}, S_{uv}$ the centred sums of $u = s$, $v = z$:

$$
\cot\theta = \frac{S_{vv} - S_{uu} + \sqrt{(S_{vv}-S_{uu})^2 + 4S_{uv}^2}}{2S_{uv}},\qquad
z_0 = \bar z - \bar s\cot\theta
$$

Then $\theta = \operatorname{atan2}(1, \cot\theta)$, $p_T = c|B_z|R/100$ (R in cm),
$q/p = q\sin\theta/p_T$, and the circle's beam distance $d_\text{beam} = \big||\mathbf{c}-\mathbf{b}| - R\big|$.

### 3.5 Hit-fit covariance ([tracks.py:209](../train/downstream/vertexing/tracks.py#L209))

Linearised least squares of the five parameters $\mathbf{a} = (l_0, l_1, \phi, \theta, q/p)$ at a reference arc
length $s_r$ ($l_0$ along $d_0$; $l_1 = z_0$, or curvilinear loc1). Per hit, with $\Delta s = s - s_r$,
left curvature $\kappa = -c\,(q/p)/\sin\theta$, turning angle $a = \kappa\Delta s$,
$k_{q/p} = c/\sin\theta$, $k_\theta = -c\,(q/p)\cos\theta/\sin^2\theta$,
$g = (1 - \operatorname{sinc}a)/a$:

| residual | $\partial/\partial l_0$ | $\partial/\partial l_1$ | $\partial/\partial\phi$ | $\partial/\partial\theta$ | $\partial/\partial(q/p)$ |
|---|---|---|---|---|---|
| normal offset $/\sigma_{r\phi}$ | $\cos a$ | 0 | $\Delta s\operatorname{sinc}a$ | $b\,k_\theta$ | $b\,k_{q/p}$ |
| $z$ $/\sigma_z$ | $-\cot\theta\sin a$ | 1 (or $1/\sin\theta$) | $-\cot\theta\,\tfrac12\Delta s^2\kappa\operatorname{sinc}^2\tfrac a2$ | $-\dfrac{\Delta s}{\sin^2\theta} + \cot\theta\,\Delta s^2 g\,k_\theta$ | $\cot\theta\,\Delta s^2 g\,k_{q/p}$ |

with $b = -\tfrac12\Delta s^2\operatorname{sinc}^2(a/2)$. Information $I = \sum_\text{hits} (\mathbf{r}_n\mathbf{r}_n^T + \mathbf{r}_z\mathbf{r}_z^T)$,
covariance $C = I^{-1}$ (hit errors only, no material). Defaults $\sigma_{r\phi} = 0.034$ cm, $\sigma_z = 0.1$ cm.

## 4. Perigee conversion ([tracks.py:259](../train/downstream/vertexing/tracks.py#L259))

Circle → perigee, with $h = \operatorname{sign}R_s$, $\mathbf{u} = (\mathbf{b} - \mathbf{c})/|\mathbf{b} - \mathbf{c}|$:

$$
d_0 = h\,(|R_s| - |\mathbf{b} - \mathbf{c}|),\qquad \phi = \operatorname{atan2}(-h u_x, h u_y)
$$

Perigee → circle (used on every route, with $B_z$ at the perigee for a field map):

$$
R_s = \operatorname{sign}(q/p)\operatorname{sign}(B_z)\frac{100\,p_T}{c|B_z|},\qquad
\mathbf{c} = \mathbf{p}_\perp + R_s(\sin\phi, -\cos\phi),\qquad p_T = \sin\theta/|q/p|
$$

Checked: a positive track in $+B_z$ moving along $+x$ bends to $-y$, centre $\mathbf{p}_\perp + R(0,-1)$, $R_s>0$.

## 5. ACTS ([perigee_propagator.cpp](../scripts/acts/perigee_propagator/perigee_propagator.cpp))

### 5.1 Start state ([tracks.py:527](../train/downstream/vertexing/tracks.py#L527))

The circle point nearest the innermost hit, $z = s\cot\theta + z_0$, the helix tangent
$\hat{\mathbf{d}} = (\sin\theta\,\hat{\mathbf{t}}_\perp, \cos\theta)$, the hit fit's $q/p$, and the hit-fit covariance
at that $s$ in curvilinear form (mm; $t$ uncorrelated, variance 1).

### 5.2 Kalman refit (`refit`, [perigee_propagator.cpp](../scripts/acts/perigee_propagator/perigee_propagator.cpp))

* Prior: the start state moved $10$ mm back along a straight line, covariance $\times 100$
  (`kalman_start_covariance_scale`).
* One plane per hit, through the hit, normal $\hat{\mathbf{n}}$ = the hit fit's tangent there; axes
  $\hat{\mathbf{u}} = \hat{\mathbf{z}}\times\hat{\mathbf{n}}/|\cdot|$, $\hat{\mathbf{v}} = \hat{\mathbf{n}}\times\hat{\mathbf{u}}$.
* Measurement $\mathbf{m} = (0,0)$ on that plane, covariance
  $V = A^T(\sigma_{r\phi}^2\hat{\mathbf{e}}_\phi\hat{\mathbf{e}}_\phi^T + \sigma_z^2\hat{\mathbf{z}}\hat{\mathbf{z}}^T)A$,
  $A = [\hat{\mathbf{u}}\ \hat{\mathbf{v}}]$, with each hit's own errors: TPC (`hit_errors_cm`) or silicon
  (`silicon_hit_errors_cm`, hits at $r <$ `max_radius`).
* Material (`acts.kalman_material`, the TPC gas): each TPC hit's plane carries a slab of gas from halfway to the
  previous TPC hit to halfway to the next (none on silicon hits); multiple scattering, and energy loss with
  `acts.energy_loss` (on, as sPHENIX fits).
* `KalmanFitter<EigenStepper, DirectNavigator>`, gain-matrix update and smoother; the smoothed state at the
  innermost measurement is kept.

### 5.3 Propagation to the perigee

Backward from that state (or from §5.1 if the refit failed) to the `PerigeeSurface` at $(b_x, b_y, 0)$, through
the tracking geometry (`Navigator`, `MaterialInteractor`: multiple scattering, energy loss per `acts.energy_loss`),
transporting the covariance. Failures (NaN) drop the track.

### 5.4 Fallback ([tracks.py](../train/downstream/vertexing/tracks.py))

Where the refit failed but propagation succeeded: $C_\text{perigee} \leftarrow s_f\,C_\text{perigee}$,
$s_f = 100$ (`acts_kalman_fallback_covariance_scale`; $\log s_f$ learnable with `learn_weights`).

## 6. Track products ([tracks.py](../train/downstream/vertexing/tracks.py))

* The vertex finder gets each track's perigee $\mathbf{q} = (d_0, z_0, \phi, \theta, q/p)$ (cm) at the beam line,
  its **full** $5\times5$ covariance $C$ (cm) $\times$ `track_error_scale`$^2$, $B_z$ at the perigee, and its hit
  counts (TPC, silicon).
* Usable: finite perigee and covariance, $p_T \ge 0.25$ GeV, $|d_0| < 1$ cm. Weight
  $w = P(\text{real})\cdot\text{usable}$ (× a softplus MLP and straight-through sigmoid gates with `learn_weights`).
  Tracks with $w > 0$ are the finder's input; $w$ scales each track's information in the vertex fit (1 for ACTS).

## 7. Primary vertices: ACTS's adaptive multi-vertex finder ([vertices.py](../train/downstream/vertexing/vertices.py))

The settings default to ACTS's `AdaptiveMultiVertexFinderAlgorithm` (the reference `compare` runs); lengths in cm.

### 7.1 Helix transport

With $\rho = \sin\theta/(q/p\cdot k B_z)$ ($k = 0.299792458\times10^{-2}$ GeV/(T·cm)), the circle centre
$\mathbf{c} = \mathbf{b} + d_0(-\sin\phi, \cos\phi) + \rho(\sin\phi, -\cos\phi)$:

* **Perigee at a point** $\mathbf{x}$ (a line along $z$ through it): $\hat{\mathbf{u}} = (\mathbf{x}_\perp - \mathbf{c})/|\cdot|$,
  PCA $= \mathbf{c} + |\rho|\hat{\mathbf{u}}$, $\phi' = \operatorname{atan2}(-\operatorname{sign}\rho\,u_x, \operatorname{sign}\rho\,u_y)$,
  $z = z_0 - \rho\operatorname{wrap}(\phi' - \phi)\cot\theta$, $d_0' = (\text{PCA} - \mathbf{x}_\perp)\cdot(-\sin\phi', \cos\phi')$,
  $z_0' = z - x_z$; $C' = JCJ^T$ with $J = \partial\mathbf{q}'/\partial\mathbf{q}$ (automatic differentiation of
  this map).
* **3D impact** at $\mathbf{x}$: Newton in the turning angle on the helix
  $\mathbf{h}(\varphi) = \mathbf{O} + \rho(-\sin\varphi, \cos\varphi, -\varphi\cot\theta)$ (20 steps) gives the 3D
  closest point $\mathbf{P}$ and direction $\hat{\mathbf{T}}$; on the curvilinear axes
  $\hat{\mathbf{U}} = \hat{\mathbf{z}}\times\hat{\mathbf{T}}/|\cdot|$, $\hat{\mathbf{V}} = \hat{\mathbf{T}}\times\hat{\mathbf{U}}$:
  $\mathbf{l} = ((\mathbf{P}-\mathbf{x})\cdot\hat{\mathbf{U}}, (\mathbf{P}-\mathbf{x})\cdot\hat{\mathbf{V}})$, covariance
  $J_lCJ_l^T$.

### 7.2 Linearisation (Piacquadio; ACTS's `HelicalTrackLinearizer`)

At a point $\mathbf{x}_0$: the perigee there $\mathbf{q}'$, PCA $\mathbf{p}$, $X = p_x - x_{0x} + \rho\sin\phi'$,
$Y = p_y - x_{0y} - \rho\cos\phi'$, $S^2 = X^2 + Y^2$, $h = \operatorname{sign}\rho$:

$$
\mathbf{q}'(\mathbf{x}, \boldsymbol{\pi}) \approx \mathbf{c} + A\mathbf{x} + B\boldsymbol{\pi},\quad \boldsymbol{\pi} = (\phi, \theta, q/p),
\qquad
A = \begin{pmatrix} -hX/S & -hY/S & 0\\ \rho\cot\theta\,Y/S^2 & -\rho\cot\theta\,X/S^2 & 1\\ -Y/S^2 & X/S^2 & 0\\ 0&0&0\\0&0&0\end{pmatrix},\
B = \begin{pmatrix} 0&0&0\\ \rho\cot\theta(1 - h\rho/S) & 0 & 0\\ h\rho/S & 0 & 0\\ 0&1&0\\0&0&1\end{pmatrix}
$$

$\mathbf{c} = \mathbf{q}' - A\mathbf{p} - B\boldsymbol{\pi}'$, $G = C'^{-1}$.

### 7.3 Kalman vertex update (Frühwirth)

With $W = (B^TGB)^{-1}$, $G_B = G - GBWB^TG$, the prior $(\mathbf{x}_p, \sigma_p^2 I)$ ($\sigma_p^2$ =
`initial_variance` = 1 cm², ACTS's $10^2$ mm²) and track weights $\omega_i$:

$$
C^{-1} = \sigma_p^{-2}I + \sum_i \omega_i A_i^TG_{B,i}A_i,\qquad
\hat{\mathbf{x}} = C\Big(\sigma_p^{-2}\mathbf{x}_p + \sum_i\omega_iA_i^TG_{B,i}(\mathbf{q}_i - \mathbf{c}_i)\Big)
$$

(the one-track-at-a-time update summed); $\chi^2 = |\hat{\mathbf{x}} - \mathbf{x}_p|^2/\sigma_p^2 + \sum_i\omega_i\mathbf{r}_i^TG_{B,i}\mathbf{r}_i$,
$\mathbf{r}_i = \mathbf{q}_i - \mathbf{c}_i - A_i\hat{\mathbf{x}}$; ndf $= 2\sum_i\omega_i - 3$. Refitted momentum
$\tilde{\boldsymbol{\pi}}_i = WB^TG\mathbf{r}_i$ (SV momenta).

### 7.4 Adaptive multi-vertex fit

Each vertex keeps its tracks' 3D impact $\mathbf{l}_i$ at its seed. Iteration $n$, temperature $T_n$ (ACTS example:
one, $T = 1$): compatibility $\chi^2_{iv} = \mathbf{r}^TW_l\mathbf{r}$, $\mathbf{r} = \mathbf{l}_i - ((\mathbf{x}_v - \mathbf{s}_v)\cdot\hat{\mathbf{U}}, (\mathbf{x}_v - \mathbf{s}_v)\cdot\hat{\mathbf{V}})$;

$$
\omega_{iv} = \frac{f_{iv}\,e^{-\chi^2_{iv}/2T}}{e^{-9/2T} + \sum_{v'} f_{iv'}\,e^{-\chi^2_{iv'}/2T}}
$$

over the vertices the track belongs to ($f$: membership, §7.5); $\omega \le 0.001$ → 0; the Kalman update (§7.3)
of every vertex; relinearise where the vertex moved $> 0.05$ cm in $x$-$y$ from its linearisation point; stop after
the last temperature once $\Delta\mathbf{x}^TC^{-1}\Delta\mathbf{x} \le 0.01$ for every vertex (≤ 30 iterations).

### 7.5 Finder loop

1. Seed: ACTS's Gaussian track density on the tracks not yet used: per track
   $\exp(c_0 + c_1z + c_2z^2)$ from its full $(d_0, z_0)$ covariance (with the correlation), $|d_0|/\sigma \le 3.5$, inside
   its window ($\chi^2 = 2\cdot 2\cdot12^2$); maximum from each track's $z_0$ with two Gaussian Newton steps, the
   first maximum in track order; the seed $(b_x, b_y, z)$.
2. Membership $f_i$ (soft): $|z_{0,i} - z_\text{seed}| < 0.1$ cm and the impact significance at the seed
   $\sqrt{(d_0'/\sigma_{d_0'})^2 + (z_0'/\sigma_{z_0'})^2} < 5$, over all input tracks. None: the seed moves to the nearest
   unused track's $z_0$; still none: stop.
3. Fit the new vertex with every kept vertex it shares tracks with (§7.4).
4. Its tracks with $\chi^2 < 18.42$ still unused leave the pool (if none: its least compatible unused track, or the
   unused track nearest in $z$). Kept if ≥ 2 such tracks, contamination $\sum\omega(1-\omega)/\sum\omega^2 \le 0.5$,
   and $|\Delta z|/\sqrt{\sigma_{z,1}^2 + \sigma_{z,2}^2} \ge 3$ to every kept vertex; else deleted, the vertices it
   was fitted with refitted without it.
5. Repeat while ≥ 2 tracks are unused (and fewer than `max_vertices` are kept).

The PV is the first vertex kept. Events with none (ACTS: no vertex) are searched again with the preselection
relaxed step by step (`fallback_preselection`: (z interval, significance) = (0.3 cm, 7), then (1 cm, 10)), each
vertex still kept by ACTS's rules; `vertex_fallback_step` says which step (0: as ACTS). Off with `hard_cuts` (ACTS)
or `pv_fallback=False`; the steps are a place for learning.

### 7.6 Soft cuts

Every cut that decides a track's membership is a sigmoid of width `cut_softness_cm` (0.005 cm),
`cut_softness_sigma` (0.25 σ) or `cut_softness_hits` (1); `hard_cuts=True` makes them steps, which is ACTS. The
vertex accept/delete decisions stay ACTS's. `compare` runs both: the hard version against ACTS AMVF event by event.

## 8. Secondary vertices

sPHENIX's `SecondaryVertexFinder` pair selection (written from its description; sPHENIX's code has no licence), then
the vertex fit of §7:

1. Candidates (soft): ≥ 20 TPC hits; displaced from the PV: with silicon hits $|d_0'|, |z_0'| \ge 0.02$ cm at the PV
   (sPHENIX); TPC-only (today's data) $\chi^2_\text{PV} > 9$ (TPC resolution is too coarse for 0.02 cm); not a track
   of another primary vertex.
2. Pairs: the $x$-$y$ intersections of their circles within $r < 40$ cm, the helix $z$ there within 1 cm, the helix
   tangents there closer than 0.5 cm, their midpoint more than 0.2 cm from the PV; the 6 with the smallest pair DCA.
3. Each seed fitted on its own (§7.4, not shared) with its pair and the candidates of $\chi^2 \le 9$ at the seed.
4. Kept if ≥ 2 compatible tracks, ndf > 0, $\chi^2/\text{ndf} \le 10$, decay length > 0.2 cm; one per group within
   0.1 cm (lowest $\chi^2/\text{ndf}$).
5. KFParticle-style outputs: momentum $\sum\tilde{\mathbf{p}}_i$ (refitted at the SV), DIRA
   $\cos\angle(\mathbf{x}_\text{SV} - \mathbf{x}_\text{PV}, \mathbf{p})$, decay length and its significance
   $L/\sqrt{\hat{\mathbf{u}}^T(C_\text{SV} + C_\text{PV})\hat{\mathbf{u}}}$, the pair DCA.
6. The PV refitted together with its SVs (§7.4, tracks shared): the SV correction to the PV.

## 9. Design notes

Status after the ACTS alignment (2026-11-07); numbers before it: 1000-event eval / 500-event compare.

* **Beam spot:** AMVF as shipped has no beam constraint (its seeds sit at $x = y = 0$); neither does VertexHead.
  The transverse residual (0.107 cm) dominates the 3D error, so a constraint is the first thing to try once the
  hard-cut check passes (an extra prior term $\Sigma_\text{beam}^{-1}$ in §7.3).
* **`track_error_scale` = 2.5:** still applied to the full covariance on both sides; the χ² cuts now act on full
  covariances, so retune after the check.
* **Full covariance:** done (§7.3; the old fit used $\sigma_{d_0}, \sigma_{z_0}$ only).
* **z window:** done (ACTS's 1 mm window around the seed plus the significance cut, then the fit's own weights).
* **Few tracks:** AMVF's events now come from the same finder; events it can't vertex take the relaxed-preselection
  fallback (§7.5). The first version (one fit on all tracks, no z window) gave 23% of the PVs with rmse 6.7 cm.
* **Kalman refit material:** done (§5.2).
* **SV geometry:** done (helix intersections and tangents, 3D impact; no straight lines at the perigee).
* **Speed:** ~18 ms/event on a laptop CPU (the Jacobians by automatic differentiation); analytic Jacobians would
  cut it.
* **Licences:** ACTS is MPL-2.0, KFParticle GPL-3.0, sPHENIX's coresoftware unlicensed: this code is written from
  the published maths and the algorithms' descriptions, and checked against ACTS's output.

### Checked and consistent

Sign conventions (§1, §4), helix transport and 3D impact (exact at a vertex: $d_0', z_0', \mathbf{l} \approx 10^{-14}$
cm), their automatic Jacobians against finite differences ($2\times10^{-8}$), gradients through the soft cuts and
weights (finite), the 10 mm straight back-off of the refit's start.
