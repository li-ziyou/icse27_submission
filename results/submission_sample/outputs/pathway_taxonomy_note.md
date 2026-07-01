# Pathway taxonomy support

## `outputs/pathway_2x2_counts.csv` and `outputs/pathway_2x2_counts.tex`
- `No human handling`: discussion=False, code_intervention=False, n=4535, pct_of_subset=21.77%.
- `Discussion only`: discussion=True, code_intervention=False, n=4473, pct_of_subset=21.48%.
- `Silent edit`: discussion=False, code_intervention=True, n=7032, pct_of_subset=33.76%.
- `Early takeover`: discussion=True, code_intervention=True, n=4788, pct_of_subset=22.99%.

The four 2x2 cells sum to 20828, exactly matching the scoped pathway-analysis subset size of 20828.
This verifies that the operational pathway taxonomy is exhaustive on the current taxonomy_df subset without changing the paper’s scoped PR subset or preprocessing.

## `outputs/pathway_first_response_by_pathway.csv`, `outputs/pathway_first_response_by_pathway.tex`, and `outputs/pathway_first_response_stats.json`
- `No human handling`: n=1, Comment-first=0.00%, Review-first=0.00%, Commit-first=100.00%.
- `Discussion only`: n=3320, Comment-first=70.48%, Review-first=28.92%, Commit-first=0.60%.
- `Silent edit`: n=1345, Comment-first=0.00%, Review-first=0.00%, Commit-first=100.00%.
- `Early takeover`: n=3638, Comment-first=63.33%, Review-first=13.58%, Commit-first=23.09%.

Chi-square=5019.9602, p=NA, Cramer's V=0.5498, N=8304.
First-response behavior is differentiated across pathways using a response-type variable that was not part of the 2x2 definition itself, which provides lightweight external empirical support for the taxonomy.
