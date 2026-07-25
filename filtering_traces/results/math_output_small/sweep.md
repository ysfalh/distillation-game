# Degeneracy filter sweep

- Traces: `math_output_small`
- Tokenizer: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `max_new_tokens`: 1024
- Conditions: `standard` (n=5000), `antidistillation_lam_0.08` (n=5000), `poe_gamma_0.75` (n=5000)

Thresholds are calibrated on the Standard teacher alone and then applied unchanged to every condition. Each row is one target acceptance rate for Standard. The realized rate is higher than the target and saturates once the target gets aggressive, because the hard loop rules and the 64-token minimum removal gate fire regardless of the calibrated thresholds.

## Dropped traces by target Standard acceptance

| target accept | kept standard | rep thr | off-script thr | standard dropped | antidistillation_lam_0.08 dropped | poe_gamma_0.75 dropped |
|---|---|---|---|---|---|---|
| 0.900 | 0.8960 | 0.0322 | 0.0073 | 520 (10.40%) | 772 (15.44%) | 454 (9.08%) |
| 0.950 | 0.9186 | 0.0522 | 0.0152 | 407 (8.14%) | 679 (13.58%) | 454 (9.08%) |
| 0.970 | 0.9408 | 0.0645 | 0.0222 | 296 (5.92%) | 622 (12.44%) | 446 (8.92%) |
| 0.980 | 0.9582 | 0.0791 | 0.0264 | 209 (4.18%) | 569 (11.38%) | 432 (8.64%) |
| 0.990 | 0.9776 | 0.1055 | 0.0365 | 112 (2.24%) | 509 (10.18%) | 411 (8.22%) |
| 0.995 | 0.9868 | 0.1309 | 0.0475 | 66 (1.32%) | 479 (9.58%) | 400 (8.00%) |
| 1.000 | 0.9950 | 0.9863 | 0.1579 | 25 (0.50%) | 431 (8.62%) | 381 (7.62%) |

## Operating point: target acceptance 0.990

| condition | n | dropped | drop rate | loop | off-language | accuracy before | accuracy after |
|---|---|---|---|---|---|---|---|
| standard | 5000 | 112 | 2.24% | 1.24% | 1.00% | 0.6116 | 0.6144 |
| antidistillation_lam_0.08 | 5000 | 509 | 10.18% | 7.32% | 3.98% | 0.5924 | 0.6143 |
| poe_gamma_0.75 | 5000 | 411 | 8.22% | 5.28% | 4.00% | 0.6116 | 0.6324 |

`loop` and `off-language` overlap, so they do not sum to the drop rate. Sampled traces from this operating point are in `trace_degenerate/`.
