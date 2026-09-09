"""AE1: measure the GRADIENT NORM each loss term puts on the CDR coord head.

Why this exists.  On 2026-09-01 loss_bond was switched on at bond_weight=1.0
and the fitted aar_cdr collapsed 0.83 -> 0.11 while bond length itself improved
6x.  The cause was not the loss VALUE but the gradient: loss_cdr is an MSE in
x0_norm space (it divides by cdr_scale**2) while loss_bond was in physical A**2,
so with s=cdr_scale=6 the bond term hit coord_head 36x harder.

Loss values cannot predict this.  A bounded term (smooth_lddt lives in [0,1] and
its sigmoids saturate) can show a big value and contribute almost no gradient,
while an unbounded quadratic can show a small value and dominate.  So before
enabling any further term, measure d(term)/d(coord_head.weight) directly.

What it prints, per term:
    |g|        L2 norm of the gradient on coord_head.weight
    ratio      |g| / |g_cdr|  -- the number that decides who wins
    w_balanced weight that would make this term's gradient equal loss_cdr's

Also serves as a check on the cdr_scale**2 change to _compute_full_bond_loss:
with the division in place, bond's ratio should drop by ~36x versus without.

Run:  python see/probe_ae1_grad_norm.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))