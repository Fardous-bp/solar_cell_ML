# Bulk–surface recombination coupling in c-Si solar cells

Code, data and figures for the manuscript
*"Regime-Dependent Coupling of Bulk and Surface Recombination in Crystalline Silicon Solar Cells:
A Physics-Based and Interpretable Machine-Learning Study"*.

## Quick start
```bash
pip install -r requirements.txt
python si_recombination_analysis.py --outdir outputs      # ~3 min on a standard CPU
```
Everything is seeded (seed = 42). One run regenerates:

| Output | Content |
|---|---|
| `outputs/data/si_cell_dataset.csv` | 23,814-device full-factorial dataset (21 τ × 21 S × 6 W × 9 N_d) |
| `outputs/results.json` | **Every number quoted in the paper** (metrics, shares, thresholds, tables) |
| `outputs/figures/fig1…fig7 (.png, .pdf)` | All figures (300 dpi PNG + vector PDF) |

## Model (all in cm, s, A)
```
1/τ_eff = 1/τ_bulk + 2S/W
J0      = q·ni²·W / (N_d·τ_eff)  =  (q·ni²/N_d)·(W/τ_bulk + 2S)      → J0 = J0_bulk + J0_surf
J(V)    = Jph − J0·[exp(qV/kT) − 1]          (ideal diode, n = 1)
Voc     = (kT/q)·ln(1 + Jph/J0)
Vmp     : exact, via the Lambert W function;   η = Vmp·Jmp / Pin
```
Fixed: T = 300 K, ni = 1.0e10 cm⁻³, Jph = 40 mA/cm², Pin = 100 mW/cm², n = 1.

## Built-in verification (asserted on every run)
* J0 additivity and Eq. (2) ≡ Eq. (2′) to < 1e-12 relative error
* Closed-form maximum power point vs. brute-force maximisation (20,001-point grid) < 1e-6
* η is a monotonically decreasing function of J0 (data collapse), up to round-off
* 0 < FF < 1 for all devices

## Reproducing the manuscript
`build_paper.js` (Node ≥ 18, `npm i docx`) reads `outputs/results.json` and `outputs/figures/*.png`
and writes `outputs/Manuscript.docx`. No number in the manuscript is typed by hand.

## Notes
* Results are bit-reproducible with the tested versions (see `requirements.txt`); other library versions
  may change last-digit values of the random-forest metrics.
* The model is intentionally idealised (no emitter/contact, Auger, radiative recombination; constant Jph).
  See Section 4 of the manuscript.
