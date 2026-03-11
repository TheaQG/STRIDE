"""
Defines explicitly which product each metric uses.

E.g.:
- probabilistc metrics -> full ensemble
- CRPS -> full ensemble
- PSD -> ensemble-member envelope and PMM
- Annual sums -> PMM, ensemble mean, and ensemble member spread
- SAL -> ensemble member spread and PMM
- Temporal persistence -> PMM and ensemble member spread
"""