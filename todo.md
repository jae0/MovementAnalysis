2. Biological Stages & Development (Thermal Degree-Days)
Larval development follows an ontogenetic thermal degree-day (DD) clock with base temperature $T_0 = -1.5^\circ\text{C}$ (or $0.0^\circ\text{C}$ depending on parameterization):
$$\Delta D = \max(0, T(z, t) - T_0) \cdot \Delta t_{\text{days}}$$

Stages:
- Zoea I: Early planktonic stage (default threshold $\approx 65\text{ DD}$).
- Zoea II: Intermediate planktonic stage (default threshold $\approx 130\text{ DD}$).
- Megalopa: Late semi-planktonic stage seeking benthic nursery grounds (default threshold $\approx 200\text{ DD}$).
- Instar I (Settled): Post-settlement juvenile crab on seabed.

Persistent Individual Traits:
- Rather than resampling random thresholds at every time step (which can cause stage-regression artifacts), each larva is assigned an immutable developmental percentile $u_{\text{dev}} \sim \mathcal{U}(0, 1)$ at initialization.
- Development uses a mean-preserving lognormal CDF schedule (MoltCDF) parameterized by cv_molt. A larva transitions once $F(D) \ge u_{\text{dev}}$, guaranteeing monotonic, non-regressing development.

3. Vertical & Horizontal Movement Dynamics
At each step, larval displacement $d\mathbf{x} = \mathbf{u}_{\text{eff}} \Delta t + \text{stochastic diffusion}$ includes:

Active Post-Hatch Ascent:
- Newly released larvae swim actively toward the surface mixed layer (target depth $z \approx -10\text{ m}$) at $w_{\text{ascent}} = w_{\text{max}} \tanh((z_{\text{target}} - z) / L_{\text{relax}})$, reflecting negative geotaxis and positive phototaxis.

Diel Vertical Migration (DVM):
- Driven by circadian cycles ($T = 86400\text{ s}$):
  $$z_{\text{target}}(t) = \bar{z} - \Delta z \cos\left(\frac{2\pi t}{86400}\right)$$
- Stage-dependent target depths:
  - Zoea I: Day $-50\text{ m}$, Night $-10\text{ m}$.
  - Zoea II: Day $-55\text{ m}$, Night $-8\text{ m}$.
  - Megalopa: Day $-120\text{ m}$, Night $-60\text{ m}$ (progressively descends toward seabed).

Passive Gravitational Sinking:
- Stage-specific settling velocities: Zoea I ($-0.5\text{ mm/s}$), Zoea II ($-1.0\text{ mm/s}$), Megalopa ($-2.5\text{ mm/s}$).

Logarithmic Bottom Boundary Layer (BBL):
- Law-of-the-wall velocity reduction factor $f_{\text{bbl}}(z) = \ln(\max(z_0, z - z_{\text{bed}}) / z_0) / \ln(h_{\text{bbl}} / z_0)$ within $h_{\text{bbl}} \approx 10\text{ m}$ of seabed, reducing horizontal advection near bottom.

Near-Surface Wave Stokes Drift:
- Coupling of wave-induced Stokes drift with exponential depth attenuation $\mathbf{u}_s(z) = \mathbf{u}_{s0} \exp(z / d_{\text{decay}})$.

Visser (1997) Diffusive Pseudo-Drift:
- Explicit $d\kappa_v/dz$ correction term preventing artificial unphysical accumulation of particles in low-diffusivity pycnoclines.

Coastline Normal Projection & Tangential Slip:
- Shoreline interactions compute the local land-sea gradient normal $\mathbf{n} = \nabla \text{land} / \|\nabla \text{land}\|$, projecting displacement alongshore ($\mathbf{t} \perp \mathbf{n}$) with tangent bisection to prevent particles from being trapped in acute coastal embayments.

4. Mortality Formulation
Instantaneous survival probability over $\Delta t_{\text{days}}$ is evaluated per stage:
$$S(t) = \exp(-M_{\text{stage}}(T) \cdot \text{frailty}_i \cdot \Delta t_{\text{days}})$$

Temperature-dependent mortality ($M_{\text{stage}}(T)$):
- Baseline rate $M_0 \approx 0.02\text{--}0.03\text{ day}^{-1}$.
- Modulated by sub-zero cold stress below $-1.5^\circ\text{C}$ and warm physiological stress initiating at $7.0^\circ\text{C}$ with an exponential penalty (approaching lethal limits at $\ge 9\text{--}10^\circ\text{C}$).

Individual Vigour (Frailty):
- Each larva holds an individual frailty factor drawn from a mean-preserving lognormal distribution ($\text{mean} = 1.0, \text{CV} = \text{cv_mortality}$).

5. Benthic Nursery Settlement (Habitat Suitability Index)
Upon reaching the competent Megalopa stage, settlement on the seafloor is determined by the nursery Habitat Suitability Index (HSI):
$$\text{HSI} = S_z(z_{\text{bed}}) \cdot S_T(T_{\text{bottom}})$$

Bathymetric depth suitability ($S_z$):
- Acceptable depth: $-250\text{ m}$ to $-50\text{ m}$.
- Optimal nursery depth: $-180\text{ m}$ to $-80\text{ m}$ ($S_z = 1.0$).

Thermal suitability ($S_T$):
- Acceptable temperature: $-1.0^\circ\text{C}$ to $6.0^\circ\text{C}$.
- Optimal Cold Intermediate Layer (CIL): $0.5^\circ\text{C}$ to $3.5^\circ\text{C}$ ($S_T = 1.0$).

Settlement Decision:
- Deterministic gate ($\text{HSI} > 0$) or stochastic individual acceptance using a Beta-distributed suitability index drawn with parameter cv_settlement.
- Once settled, the larva transitions to :instar1_settled and its position is anchored to $(x, y, z_{\text{bed}})$.



# notes on parameterizations 

 
### 1. Thermal Degree-Days & Ontogeny

| Parameter / Feature | Modeled Value | Empirical / Biological Benchmark | Assessment |
| :--- | :--- | :--- | :--- |
| **Base temperature ($T_0$)** | $-1.5^\circ\text{C}$ (or $0.0^\circ\text{C}$) | $-1.5^\circ\text{C}$ (Kuhn & Choi 2011, Sainte-Marie 1999) | **Correct & Sensible.** Sub-zero embryonic and larval development occurs down to freezing in the Cold Intermediate Layer (CIL). Using $T_0 = -1.5^\circ\text{C}$ avoids truncation artifacts in the $-1^\circ\text{C} \le T \le 0^\circ\text{C}$ window. |
| **Zoea I $\to$ Zoea II** | $65\text{ DD}$ | $\sim 50\text{--}70\text{ DD}$ | **Consistent.** At $4^\circ\text{C}$ ($5.5^\circ\text{C}$ above $T_0$), this predicts $\sim 12\text{ days}$; at $1.5^\circ\text{C}$ ($3.0^\circ\text{C}$ above $T_0$), $\sim 21\text{ days}$. Matches Webb et al. (2007) and Incze et al. (1987). |
| **Zoea II $\to$ Megalopa** | $130\text{ DD}$ cumulative ($\Delta = 65\text{ DD}$) | $\sim 120\text{--}150\text{ DD}$ cumulative | **Consistent.** Predicts similar or slightly longer duration than Zoea I. |
| **Megalopa $\to$ Settle** | $200\text{ DD}$ cumulative ($\Delta = 70\text{ DD}$) | $\sim 190\text{--}230\text{ DD}$ cumulative | **Consistent.** Total pelagic larval duration (PLD) over average Scotian Shelf summer surface/CIL profiles ($\sim 2\text{--}4^\circ\text{C}$) equates to $\sim 45\text{--}65\text{ days}$, closely aligning with observed spring hatch (April/May) to summer settlement (July/August). |
| **Dispersal & Traits** | Lognormal CDF schedule + fixed quantile $u_{\text{dev}}$ | Individual variability in moulting | **Robust.** Fixes the historical bug where resampling per timestep caused reverse ontogeny or flickering competence. |

---

### 2. Vertical & Horizontal Movement Dynamics

| Process | Parameter / Setting | Empirical / Physical Literature | Assessment |
| :--- | :--- | :--- | :--- |
| **Active Ascent** | $w_{\text{ascent}} \le 10\text{ mm/s}$ ($0.010\text{ m/s}$), target $-10\text{ m}$ | Crab zoea upward swimming speeds: $5\text{--}15\text{ mm/s}$ (Forward 1988, Epifanio 2016) | **Sensible.** For a $150\text{ m}$ water column, ascent takes $\approx 4\text{ hours}$, rapidly placing newly hatched larvae into the euphotic layer during early spring. |
| **DVM: Zoea I & II** | Night: $-10\text{ m}$ / $-8\text{ m}$<br>Day: $-50\text{ m}$ / $-55\text{ m}$ | Plankton surveys in Baie Sainte-Marguerite & Bering Sea (Lovrich et al. 1995, Incze et al. 1987) | **Accurate.** Early zoeae track the warm surface layer at night for feeding/development and descend below the thermocline into the upper CIL during the day to avoid visual predators. |
| **DVM: Megalopa** | Night: $-60\text{ m}$<br>Day: $-120\text{ m}$ | Lovrich et al. (1995) | **Accurate.** Megalopae become semi-benthic, seeking deep shelf depressions and nursery habitat. |
| **Swimming Speeds** | $w_{\max} \approx 5\text{ mm/s}$ ($0.005\text{ m/s}$) | Zoea swimming: $3\text{--}8\text{ mm/s}$; Megalopa: $10\text{--}20\text{ mm/s}$ | **Sensible.** Migration over $\Delta z = 40\text{ m}$ takes $\sim 2.2\text{ hours}$, easily completed during twilight transitions. |
| **Passive Sinking** | Zoea I: $-0.5\text{ mm/s}$<br>Zoea II: $-1.0\text{ mm/s}$<br>Megalopa: $-2.5\text{ mm/s}$ | Body excess density ($\Delta \rho \approx 15\text{--}25\text{ kg/m}^3$) + gravitational settling (Sulkin 1984) | **Physically Sound.** Sinking speeds increase with larval carapace mass and calcification. |
| **Logarithmic BBL** | $h_{\text{bbl}} = 10\text{ m}, z_0 = 1\text{ mm}$ | Law of the wall for shelf boundary layers | **Standard.** Accurately prevents high slip velocities near the seabed. |
| **Stokes Drift** | Exponential decay with depth ($d_{\text{decay}} \sim 10\text{ m}$) | Phillips (1977), Kenyon (1969) | **Physically Sound.** Confined to the upper $10\text{--}20\text{ m}$, affecting larvae only during nighttime surface occupation. |
| **Visser (1997) Drift** | Vertical pseudo-drift $d\kappa_v/dz$ | Visser (1997) *MEPS* | **Necessary.** Prevents numerical particle trapping inside sharp pycnoclines. |
| **Coastline Normal Slip** | Tangential projection along local shoreline normal $\mathbf{n}$ | Hydrodynamic boundary condition | **Robust.** Resolves the issue of acute coastal embayment trapping. |

---

### 3. Mortality Formulation

| Component | Setting in Code | Empirical Benchmark | Notes / Discrepancies |
| :--- | :--- | :--- | :--- |
| **Base Rate ($M_0$)** | $0.02\text{--}0.03\text{ day}^{-1}$ | Pelagic larval mortality: $0.02\text{--}0.08\text{ day}^{-1}$ (Rumrill 1990) | **Sensible.** Over a 50-day PLD at base temperature, $S = \exp(-0.02 \times 50) \approx 37\%$, providing realistic baseline recruitment before thermal stress and advective losses. |
| **Thermal Thresholds** | $T_{\text{warm,crit}} = 7.0^\circ\text{C}$<br>$T_{\text{cold,crit}} = -1.5^\circ\text{C}$ | Sub-lethal stress at $\ge 7^\circ\text{C}$; lethal at $\sim 9\text{--}10^\circ\text{C}$ (Kuhn & Choi 2011) | **Accurate.** Note: the text in `todo.md` mentions *"stress above $10^\circ\text{C}$"*, whereas the actual code in [`larval_thermal_mortality_rate`](file:///c:/home/jae/projects/ParticleTracking/src/biology/larval_behavior.jl#L2255) uses $T_{\text{warm,crit}} = 7.0^\circ\text{C}$. The $7.0^\circ\text{C}$ threshold in code is biologically better supported than $10^\circ\text{C}$ because snow crab larvae show elevated mortality and metabolic distress well before $10^\circ\text{C}$. |
| **Individual Frailty** | Lognormal frailty multiplier (mean 1.0, CV = `cv_mortality`) | Proportional hazards / unobserved heterogeneity | **Theoretically Sound.** Preserves population mean while avoiding instantaneous mass extinction. |

---

### 4. Benthic Nursery Settlement (HSI)

| Criteria | Parameterization | Scotian Shelf / St. Lawrence Field Observations | Assessment |
| :--- | :--- | :--- | :--- |
| **Depth Bounds** | Acceptable: $-250\text{ m}$ to $-50\text{ m}$<br>Optimal: $-180\text{ m}$ to $-80\text{ m}$ | Dionne et al. (2003), Sainte-Marie et al. (1999), Choi & Zisserson (2012) | **Accurate.** Snow crab instars and juveniles on the Scotian Shelf concentrate in middle shelf basins and banks between 80 m and 180 m. Waters $<50\text{ m}$ are subject to storm wave disturbance and summer warming; depths $>250\text{ m}$ encounter warm Slope Water. |
| **Bottom Temp** | Acceptable: $-1.0^\circ\text{C}$ to $6.0^\circ\text{C}$<br>Optimal: $0.5^\circ\text{C}$ to $3.5^\circ\text{C}$ | Tremblay (1997), DFO Snow Crab Survey Reports | **Accurate.** $0.5^\circ\text{C}$ to $3.5^\circ\text{C}$ defines the core CIL nursery footprint. Temperatures $>6^\circ\text{C}$ are inhospitable to early juvenile instars. |
| **Beta Perturbation** | `draw_beta_index` on log-odds / Beta concentration | Bounded stochastic index $\in [0, 1]$ | **Correct.** Avoids clipping artifacts that artificially suppress mean settlement rates. |
