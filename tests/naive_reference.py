# -*- coding: utf-8 -*-
"""An independent, deliberately naive reference for the DWN forward pass.

Written from docs/checkpoint-format.md rather than from extract.py, and structured to share as
little as possible with it: explicit per-node loops, explicit bit shifts, explicit group slices,
an explicit argmax scan. Vectorised code and loop code fail differently, which is the point.
"""
import numpy as np


def naive_lut_layer(x_bits, tables, wiring):
    n_samples = x_bits.shape[0]
    out_size, n = wiring.shape
    out = np.zeros((n_samples, out_size), dtype=bool)
    for s in range(n_samples):
        for j in range(out_size):
            addr = 0
            for l in range(n):                       # §2: slot l is address bit l, LSB first
                if x_bits[s, wiring[j, l]]:
                    addr += (1 << l)
            out[s, j] = bool(tables[j, addr])        # §1: the table bit
    return out


def naive_group_sum_argmax(x_bits, num_classes):
    n_samples, width = x_bits.shape
    group = width // num_classes                     # §4: contiguous, in order
    scores = np.zeros((n_samples, num_classes), dtype=np.int64)
    winners = np.zeros(n_samples, dtype=np.int64)
    for s in range(n_samples):
        for c in range(num_classes):
            total = 0
            for b in range(c * group, (c + 1) * group):
                total += int(x_bits[s, b])
            scores[s, c] = total
        best, best_c = -1, 0
        for c in range(num_classes):                 # strict >: ties keep the LOWEST index
            if scores[s, c] > best:
                best, best_c = scores[s, c], c
        winners[s] = best_c
    return winners, scores


def naive_forward(x_bits, layers, num_classes):
    for tables, wiring, _kind in layers:
        x_bits = naive_lut_layer(x_bits, tables, wiring)
    return naive_group_sum_argmax(x_bits, num_classes)


def naive_encode(xq, thr_q):
    """§ the encoder: one bit per threshold, strict >, feature-major then threshold."""
    n_samples, n_features = xq.shape
    z = thr_q.shape[1]
    out = np.zeros((n_samples, n_features * z), dtype=bool)
    for s in range(n_samples):
        for f in range(n_features):
            for b in range(z):
                out[s, f * z + b] = xq[s, f] > thr_q[f, b]
    return out
