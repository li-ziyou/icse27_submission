The pathway taxonomy can be described in the Methodology as a study-derived 2x2 operational taxonomy over the scoped agent-PR timeline: visible human discussion (any human comment or review) crossed with visible human code intervention (any human commit). The exhaustive cell counts in `outputs/pathway_2x2_counts.tex` sum to 20828, matching the current pathway-analysis subset exactly and showing that the operationalization covers the full scoped sample without changing the main preprocessing pipeline.

The seniority score can be described in the Methodology as a study-defined proxy built from public GitHub account tenure and public contribution counts, with tenure and contribution terms transformed via log1p, 5th–95th percentile winsorization, and min-max normalization before weighting. Robustness was checked over five alternative weighting specifications using the same profiled human-developer subset and the same earliest-visible-human-entrant PR labeling rule; see `outputs/seniority_robustness_summary.tex`.

In the Results, the seniority conclusion is stable across all five specifications: labeled PR coverage ranges from 1377 to 1377, expert Discussion only + Early takeover ranges from 92.70% to 95.64%, novice Silent edit ranges from 8.98% to 10.24%, and mid Silent edit ranges from 9.06% to 11.23% (Cramer's V 0.0440 to 0.0927). Under the paper’s default 0.35/0.65 weighting (`w35_65`), the corresponding values are 94.63%, 9.93%, and 10.04%, with 1377 labeled PRs and Cramer's V=0.0604.

In the Results, the pathway taxonomy is also behaviorally differentiated by first human response type on the responded subset with pathway labels (`outputs/pathway_first_response_by_pathway.tex`; N=8304). Discussion only is 70.48% Comment-first, Silent edit is 100.00% Commit-first, and Early takeover is split across discussion and code responses (63.33% Comment-first, 13.58% Review-first, 23.09% Commit-first); chi-square=5019.9602, p=NA, Cramer's V=0.5498.

Table references and key numbers:
- `outputs/seniority_robustness_summary.tex`: default `w35_65` row uses 1377 labeled PRs with Cramer's V=0.0604.
- `outputs/pathway_2x2_counts.tex`: exhaustive 2x2 pathway counts sum to 20828.
- `outputs/pathway_first_response_by_pathway.tex`: pathway x first-response association uses N=8304 and Cramer's V=0.5498.
