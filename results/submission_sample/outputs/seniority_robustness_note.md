# Seniority robustness

## `outputs/seniority_robustness_summary.csv` and `outputs/seniority_robustness_summary.tex`
- `contrib_only`: labeled_prs=1377, chi2=5.3391, p=NA, Cramer's V=0.0440, expert Discussion only + Early takeover=92.70%, novice Silent edit=8.98%, mid Silent edit=9.06%.
- `tenure_only`: labeled_prs=1377, chi2=21.1340, p=NA, Cramer's V=0.0876, expert Discussion only + Early takeover=95.30%, novice Silent edit=9.87%, mid Silent edit=10.80%.
- `w35_65`: labeled_prs=1377, chi2=10.0510, p=NA, Cramer's V=0.0604, expert Discussion only + Early takeover=94.63%, novice Silent edit=9.93%, mid Silent edit=10.04%.
- `w50_50`: labeled_prs=1377, chi2=23.6644, p=NA, Cramer's V=0.0927, expert Discussion only + Early takeover=95.26%, novice Silent edit=10.24%, mid Silent edit=10.44%.
- `w65_35`: labeled_prs=1377, chi2=22.5629, p=NA, Cramer's V=0.0905, expert Discussion only + Early takeover=95.64%, novice Silent edit=9.13%, mid Silent edit=11.23%.

The directional result is stable across all five weighting choices: expert entrants remain concentrated in Discussion only plus Early takeover (92.70% to 95.64%), while Silent edit remains concentrated among novice and mid entrants (novice 8.98% to 10.24%; mid 9.06% to 11.23%).
Association strength stays in the same small-to-moderate band across specifications (Cramer's V 0.0440 to 0.0927), so the substantive conclusion does not depend on the exact tenure/contribution weighting.

## `outputs/seniority_robustness_full_tables.csv`
- Rows=60 across 5 specifications x 3 seniority strata x 4 pathways; each row reports the exact count, row total, and row percentage used for the summary table.
- `No human handling` is 0 in every labeled contingency table because PR-level seniority requires a visible human entrant after PR open.
The full tables preserve the original agent-PR pathway subset and only swap the weighting used to assign developer strata, which keeps this check lightweight and directly publication-oriented.
