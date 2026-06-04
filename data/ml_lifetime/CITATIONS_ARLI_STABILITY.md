# Citation Database — ArLi Stability ML Model (v4.7+)

All references that **must** be cited in any publication using v4.7 lit-anchored model.

---

## A. **PRIMARY anchors** for v4.7 inert subclass parameters

### Decay kinetics (Ea_d, lnA_d)

**[1] Stanetty, P.; Mihovilovic, M. D.** *J. Org. Chem.* **1997**, *62*, 1514–1515.
- DOI: **10.1021/jo961701a**
- Title: *"Half-Lives of Organolithium Reagents in Common Ethereal Solvents"*
- Key data extracted:
  - n-BuLi/THF: Ea_d = 75.7 kJ/mol, lnA_d = 21.9 (multi-T Arrhenius from t½ at 0, +20°C)
  - n-BuLi/THP: Ea_d = 75.5 kJ/mol, lnA_d = 19.4 (+20, +35°C)
  - s-BuLi/Et2O: Ea_d = 61.6 kJ/mol, lnA_d = 17.8 (-20, 0°C)
  - t-BuLi/THF: Ea_d = 51.2 kJ/mol, lnA_d = 16.1 (-40, -20°C)
- Role in v4.7: Primary proto-de-Li-by-THF mechanism anchor

**[2] Honeycutt, S. C.** *J. Organomet. Chem.* **1971**, *29*, 1–5.
- Title: *"The kinetics of the cleavage of tetrahydrofuran by n-butyllithium"*
- DOI: 10.1016/S0022-328X(00)82905-X (approximate)
- Key data:
  - n-BuLi/Et2O: Ea_d = 79.9 kJ/mol, lnA_d = 19.2 (+20, +35°C)
- Role: Cross-validation of Stanetty Ea_d for Et2O-class solvents

**[3] Fitt, J. J.; Gschwend, H. W.** *J. Org. Chem.* **1984**, *49*, 209–210.
- DOI: **10.1021/jo00175a046**
- Title: (DME half-lives)
- Key data:
  - t-BuLi/DME at -70°C: t½ = 11 min
  - s-BuLi/DME at -70°C: t½ = 120 min
  - n-BuLi/DME at -20°C: t½ = 111 min
- Role: Single-T anchors for DME (extended solvent coverage)

### Formation kinetics (Ea_f, lnA_f)

**[4] Charton, M.** in *"The Chemistry of Organolithium Compounds"* (Patai/Rappoport), Wiley, **2004**, Ch. 7 §VI.A.2, p. 296.
- Editors: Z. Rappoport, I. Marek
- ISBN: 0-470-84339-X
- Cites: **Batalov, A. P.; Rostokin, V. I.** (Ref 65 in Charton's chapter)
- Key equation (eq. 36):
  - `log k(X-C6H4-Br + PhLi) = 5.07·σ_X − 2080/T + 6.84` (R² = 95.55%, n=15)
  - → Ea_f(ArLi + ArBr exchange) = 40 kJ/mol
  - → lnA_f baseline (σ=0) = 15.8
  - → Hammett ρ_f = 5.07 (EWG strongly accelerates exchange)
- Role: Primary Ea_f anchor + Hammett-σ-dependent rate law

**[5] Leroux, F.; Schlosser, M.; Zohar, E.; Marek, I.** in same volume, Ch. 9, p. 435–494.
- Title: *"The preparation of organolithium reagents and intermediates"*
- Key qualitative anchor (p. 440):
  - "PhBr + n-BuLi/THF/-75°C: complete in a few seconds" → k ≈ 1 s⁻¹
  - "Iodobenzene reacts instantaneously"
- Role: Calibrates n-BuLi vs PhLi nucleophile difference → adjusts (40, 15.8) → (32, 21) for n-BuLi/ArBr

---

## B. **STRUCTURAL & MECHANISTIC** references (cite for chemistry framing)

**[6] Reich, H. J.** *Chem. Rev.* **2013**, *113*, 7130–7178.
- DOI: **10.1021/cr400156x**
- Title: *"Role of Organolithium Aggregates and Mixed Aggregates in Organolithium Mechanisms"*
- Use: General mechanism discussion, aggregation states, p. 7164 has Reich's own Li/Br exchange ΔG‡ data (PhLi + 4-Me-PhBr/Et2O: ΔG‡ = 23.3 kcal/mol, ΔS‡ = -23.6 eu)

**[7] Reich, H. J.; Green, D. P.; Medina, M. A.; Goldenberg, W. S.; Gudmundsson, B. Ö.; Dykstra, R. R.; Phillips, N. H.** *J. Am. Chem. Soc.* **1998**, *120*, 7201–7210.
- Title: *"Aggregation and Reactivity of Phenyllithium Solutions"*
- Use: Aggregation states of PhLi in THF/Et2O (dimer/monomer equilibrium ΔG‡ ≈ 8 kcal/mol)

**[7c] Reich, H. J.** *Chem. Rev.* **2013**, *113*, 7130–7178 (revisited for v6.0).
- Key data point for v6.0 lnA prior:
  - ΔS‡ ≈ -32 eu (cal·mol⁻¹·K⁻¹) for n-BuLi + Ph₃CH/THF
  - → lnA ≈ ln(k_B·T/h) + ΔS‡/R + 1 ≈ 22 ± 5 at 298 K
- Used in v6.0 Bayesian model as prior for all class-level lnA_d
- Validates conclusion that ArLi lnA is dominated by aggregation/THF rearrangement entropy, not TS configurational entropy alone

**[7b] Luisi, R.; Capriati, V. (Eds.)** *Lithium Compounds in Organic Synthesis: From Fundamentals to Applications*, Wiley-VCH, **2014**.
- ISBN: 978-3-527-33343-5
- Key chapters cited:
  - **Ch. 18, Table 18.3** — PhLi/THF half-life ≈ 100 h at +20°C (anchors high-end of inert ArLi stability)
  - Ch. 17 — aryllithium intramolecular cyclization (5-exo benzyne / cyano attack) mechanism
  - Ch. 19 — ortho-functionalized ArLi (chelation + steric)
- Role: Secondary cross-check anchor for v4.7 (Ea_d=78 lit anchor)

---

## B'. **v5.0 framework references** (informative for v5.0 negative-result discussion)

**[Charton-LFER]** Charton, M. *Prog. Phys. Org. Chem.* (various, 1981–2004) — Hammett σ + Taft Es tabulated values used in v5.0 SIGMA_TABLE for 22 substituents.

**[Hansch-Leo]** Hansch, C.; Leo, A. *Exploring QSAR — Fundamentals and Applications in Chemistry and Biology*, ACS, **1995** — supplementary σ_p, σ_m, σ_o values.

**[v5.0 attempt]** This work, MODEL_CHANGELOG §"v5.0 attempt" — unified Hammett-LFER framework attempted on n=24 reactive ArLi Tier B+; all decay parameters give LOO R² < 0 (NEGATIVE result). Only **m-ArLi formation** Hammett correlation (Ea_f, lnA_f vs σ_m) was positive with LOO R² = 0.47, 0.87 respectively. Documented as informative negative + partial finding; **does not replace v4.7** as paper recommendation.

---

## C. **In-house** training data (cite as our own work)

**[8] Zhao, W.; ... (this work)** — `global_arrhenius.csv` Tier 1 dataset (n=7 p-ArLi, n=5 m-ArLi, n=7 o-ArLi)
- p-OMe-PhLi: Ea_d = 79.58 kJ/mol (Tier 1, n_T = 7, ROH quench)
- p-CN-PhLi: Ea_d = 28.83 kJ/mol
- (etc.)
- v4.6 formulas in `class_fitted_models_v46.csv`

---

## D. **Yoshida-style flow chemistry** (for substrate-scope context)

**[9] Yoshida, J.; Nagaki, A.** *Chem. Eur. J.* **2008**, *14*, 7450.
- DOI: **10.1002/chem.200800582**
- Title: *"Flash Chemistry: Fast Chemical Synthesis by Using Microreactors"*
- Use: Generic flow chemistry framework + ArLi flow generation methodology

**[10] Yoshida, J.; Nagaki, A.; Yamada, T.** *Angew. Chem. Int. Ed.* **2010**, *49*, 4017.
- DOI: **10.1002/anie.200906317**
- Title: *"Flash Chemistry: New Strategies for Highly Selective Reactions"*
- Use: ArLi stability data context

---

## E. **TO BE FOUND** (priority literature mining targets)

- **Knochel, P.** *Acc. Chem. Res.* **2008**, *41*, 1404 (TMPMgCl/ArLi t½ at -40, -78°C)
- **Schlosser, M.** various 1985-2005 (ArLi superbase + stability)
- **Mongin, F.; Chevallier, F.** review (Rennes group, lithiation kinetics)
- **Collum, D. B.** various JACS papers (mathematical mechanism)
- **Bauer, W.; Winchester, W. R.; Schleyer, P. v. R.** *Organometallics* **1987**, *6*, 2371 (NMR + colligative properties of PhLi/n-BuLi)

---

## Suggested boilerplate citation paragraph for paper Methods section

```
v4.7 inert-subclass formation parameters (Ea_f = 32 kJ/mol, lnA_f = 21)
were derived by combining the Hammett-corrected ArLi/ArBr halogen-metal
exchange rate law of Batalov and Rostokin (as compiled in Charton's
review [4]) with the qualitative kinetic observations of Leroux,
Schlosser, Zohar and Marek [5] for n-BuLi as the nucleophile.

Decay parameters (Ea_d = 78 kJ/mol, lnA_d = 22) were obtained as the
weighted mean of multi-temperature half-life measurements by Stanetty &
Mihovilovic [1] (n-BuLi/THF), Honeycutt [2] (n-BuLi/Et2O), and our
in-house anchor for p-methoxyphenyllithium (Tier 1, n_T = 7).

The same anchor set was applied uniformly to all three positional
subclasses (p-, m-, o-ArLi), reflecting the substituent-independence of
the proto-de-Li-by-THF-α-H decomposition pathway in the absence of an
intramolecular attack target.
```
