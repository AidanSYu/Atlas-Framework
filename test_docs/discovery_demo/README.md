# Discovery Demo Corpus

This folder contains a realistic mock document set for Discovery Engine demos.

## Upload Set

Upload the `.txt` files in this folder to a single project before starting a Discovery session.

## Demo Goal

Use a prompt like:

`Find orally bioavailable KRAS G12C inhibitor candidates with improved safety over first-pass scaffolds.`

The corpus is designed to drive coordinator extraction of:

- domain (`organic_chemistry`)
- target (`KRAS G12C`, switch-II pocket)
- constraints (MW, LogP, TPSA, hERG, CYP3A4, solubility)
- exclusions (naphthalene-heavy cores, aniline nitro motifs)
- success metrics (biochemical IC50 and cellular EC50 thresholds)

## Files

- `target_brief_kras_g12c.txt`
- `prior_campaign_notes.txt`
- `admet_alerts_and_exclusions.txt`
- `medicinal_chemistry_hypotheses.txt`
- `assay_protocol_summary.txt`
- `synthesis_and_supply_constraints.txt`
- `literature_signals.txt`
- `session_start_prompt_examples.txt`

