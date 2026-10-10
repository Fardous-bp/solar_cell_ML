Bulk–Surface Recombination Coupling in Crystalline Silicon Solar Cells

Code and computational data accompanying the study:

“Regime-Dependent Coupling of Bulk and Surface Recombination in Crystalline Silicon Solar Cells: A Physics-Based and Interpretable Machine-Learning Study”

This repository contains the numerical implementation, generated dataset, analysis results, and reproducibility information for a physics-based study of bulk and surface recombination losses in crystalline silicon (c-Si) solar cells.

The analysis combines an analytical single-diode model with interpretable machine learning to investigate how bulk lifetime, surface recombination velocity, wafer thickness, and base doping influence device performance.

---

## Repository Contents

```text
solar_cell_ML/
├── README.md
├── LICENSE
├── requirements.txt
├── results.json
└── si_recombination_analysis.ipynb
```

### Main files

* **`si_recombination_analysis.ipynb`**
  Main analysis script. Generates the full-factorial dataset, calculates device metrics, performs analytical decomposition, trains and evaluates the machine-learning surrogate, and generates the reported numerical results and figures.

* **`results.json`**
  Machine-readable collection of calculated results, including validation metrics, variance decompositions, regime analysis, efficiency thresholds, sensitivity analyses, and other reported quantities.

* **`requirements.txt`**
  Python package dependencies required to reproduce the computational analysis.

* **`README.md`**
  Documentation and instructions for reproducing the computational analysis.

* **`LICENSE`**
  MIT License.

---

## Computational Model

The study uses a one-dimensional, base-dominated, recombination-lumped model for a crystalline silicon solar cell with symmetric surfaces.

The effective minority-carrier lifetime is written as

```text
1/τ_eff = 1/τ_bulk + 2S/W
```

where:

* `τ_bulk` = bulk minority-carrier lifetime
* `S` = surface recombination velocity
* `W` = wafer thickness

The dark saturation current density is written as

```text
J0 = q·ni²·W / (Nd·τ_eff)
```

or equivalently,

```text
J0 = J0_bulk + J0_surf

J0_bulk = (q·ni²/Nd)·(W/τ_bulk)

J0_surf = (q·ni²/Nd)·2S
```

The device current-voltage relation is based on an ideal single-diode equation:

```text
J(V) = Jph − J0[exp(qV/kT) − 1]
```

with ideality factor:

```text
n = 1
```

The maximum-power-point quantities are calculated analytically using the Lambert W function.

---

## Parameter Space

A full-factorial dataset containing **23,814 devices** is generated from:

| Parameter                           |     Range / Values | Number of levels |
| ----------------------------------- | -----------------: | ---------------: |
| Bulk lifetime, `τ_bulk`             |     `1 µs – 10 ms` |               21 |
| Surface recombination velocity, `S` |     `1 – 10⁴ cm/s` |               21 |
| Wafer thickness, `W`                |      `50 – 300 µm` |                6 |
| Base doping, `Nd`                   | `10¹⁵ – 10¹⁷ cm⁻³` |                9 |

Thus:

```text
21 × 21 × 6 × 9 = 23,814 devices
```

The lifetime, surface recombination velocity, and doping grids are logarithmically spaced, while wafer thickness is sampled linearly.

---

## Fixed Device Parameters

Unless otherwise specified:

```text
Temperature, T                  = 300 K
Intrinsic carrier density, ni  = 1.0 × 10¹⁰ cm⁻³
Photocurrent density, Jph      = 40 mA/cm²
Incident power density, Pin    = 100 mW/cm²
Ideality factor, n             = 1
```

---

## Analysis Components

The computational workflow includes:

### 1. Analytical device calculation

For every parameter combination, the script calculates:

* Effective lifetime
* Bulk recombination contribution
* Surface recombination contribution
* Total `J0`
* Open-circuit voltage
* Maximum-power-point voltage
* Maximum-power-point current
* Fill factor
* Conversion efficiency
* Surface recombination fraction
* Recombination regime
* Efficiency-target classification

### 2. Physics-based decomposition

The analysis evaluates the additive structure of `J0` and examines the resulting efficiency response.

The bulk/surface crossover occurs when:

```text
J0_bulk = J0_surf
```

giving:

```text
S = W / (2τ_bulk)
```

This provides the basis for identifying bulk-limited, surface-limited, and mixed recombination regimes.

### 3. Full-factorial variance decomposition

Because the parameter space is a balanced full-factorial design, functional ANOVA is used to quantify:

* First-order contributions
* Pairwise interaction contributions
* Total effects

### 4. Machine-learning surrogate

A Random Forest regression model is used as an interpretable surrogate for device efficiency.

The analysis includes:

* Random train/test validation
* Interleaved parameter-level holdouts
* Bulk-lifetime extrapolation
* Surface-recombination extrapolation
* Permutation importance
* Partial dependence analysis
* Individual conditional expectation analysis

The machine-learning analysis is used to examine whether the known physical structure can be recovered from a data-driven surrogate.

### 5. Physics-based baseline

An isotonic regression model using `log10(J0)` is included as a physics-informed baseline.

### 6. Sensitivity analysis

An additional Auger-recombination sensitivity analysis examines how the main qualitative conclusions respond to an additional bulk-recombination mechanism.

This analysis is intended as a sensitivity bracket rather than a calibrated full device model.

---

## Built-in Verification

The script performs numerical consistency checks during execution, including:

* Equivalence of the two expressions for `J0`
* Bulk/surface `J0` additivity
* Analytical versus brute-force maximum-power-point calculation
* Monotonicity of efficiency with respect to `J0`
* Physical bounds on fill factor
* Closure of the functional ANOVA decomposition

---

## Reproducing the Analysis

### Requirements

The analysis was developed and tested using:

```text
Python 3.12.3
NumPy 2.4.4
SciPy 1.17.1
scikit-learn 1.8.0
Matplotlib 3.10.8
```

The required Python packages are listed in:

```text
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

### Run

Execute:

```bash
python si_recombination_analysis.py
```

or, if using the output-directory option:

```bash
python si_recombination_analysis.py --outdir outputs
```

The script performs the complete computational workflow and generates the analysis outputs.

The complete calculation takes approximately a few minutes on a standard desktop CPU.

---

## Reproducibility

The computational analysis uses a fixed random seed:

```text
seed = 42
```

The full-factorial dataset is generated deterministically from the parameter grids.

Random-forest results are therefore reproducible when the same software environment and random seed are used.

The repository records the Python dependencies in `requirements.txt` to facilitate reproduction of the computational environment.

---

## Scope and Model Limitations

The model is intentionally simplified to isolate the relationship between bulk and surface recombination.

It does **not** explicitly include:

* Emitter recombination
* Contact recombination
* Series resistance
* Shunt resistance
* Radiative recombination
* Injection-dependent lifetime
* Doping-dependent lifetime
* Band-gap narrowing
* A fully calibrated Auger model
* Spatially resolved minority-carrier transport
* Experimentally measured device parameters

The effective-lifetime formulation is therefore a reduced-order representation rather than a replacement for a spatially resolved semiconductor device model.

The uniform excess-carrier approximation has a restricted physical validity range at high surface recombination velocity. The analysis therefore distinguishes the mathematical behavior of the adopted model from the regime in which the underlying approximation is expected to be physically reliable.

The reported absolute efficiencies should consequently be interpreted as **recombination-limited model results**, rather than predictions of complete experimental silicon solar-cell performance.

---

## Data and Computational Transparency

All numerical quantities reported in the study are generated from the equations and parameter ranges implemented in:

```text
si_recombination_analysis.py
```

The repository is intended to allow independent users to:

1. Recreate the parameter grid.
2. Recalculate the device metrics.
3. Verify the analytical relationships.
4. Reproduce the statistical and machine-learning analyses.
5. Inspect the numerical results independently.

No experimental dataset is used as the primary source of the computational results.

---

## Citation

If you use this code, dataset, or analysis framework in your research, please cite the associated publication:

> Fardous Hasan Bappy et al.,
> “Regime-Dependent Coupling of Bulk and Surface Recombination in Crystalline Silicon Solar Cells: A Physics-Based and Interpretable Machine-Learning Study.”

The final bibliographic information should be updated here after publication.

---

## License

This repository is released under the **MIT License**.

See `LICENSE` for details.
