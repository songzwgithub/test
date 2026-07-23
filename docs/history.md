# Hengshui L01028 Release History

Historical code versions are preserved by Git history and the intended tag `pre-l01028-release-consolidation`, not by active executable legacy source.

Key scientific changes:

- Old V2 used an unbounded `exp(Ske)` spatial model and failed under out-of-domain spatial extrapolation.
- The accepted release uses bounded Ske with `Ske_min=1e-8`, `Ske_max=0.05`, G0 no geology, shared confined lag, RBF dimension 24, and lambda 30.
- The unconfined lag is fixed at 10 days because it is weakly practically identifiable.
- The storage delayed-response sign was corrected so positive lag means `y(t-lag)`, giving a positive delayed peak shift of 55.77321162652652 days.
- Storage uncertainty is reported as a 95% structural amplitude envelope, not a full probabilistic confidence or credible interval.

No old executable source is retained in the active release tree.
