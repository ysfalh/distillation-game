# Degeneracy filter sweep

- Traces: `gsm8k_output_small`
- Tokenizer: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `max_new_tokens`: 1024
- Conditions: `standard` (n=5238), `antidistillation_lam_0.055` (n=5238), `poe_gamma_0.7` (n=5238)

Thresholds are calibrated on the Standard teacher alone and then applied unchanged to every condition. Each row is one target acceptance rate for Standard. The realized rate is higher than the target and saturates once the target gets aggressive, because the hard loop rules and the 64-token minimum removal gate fire regardless of the calibrated thresholds.

## Dropped traces by target Standard acceptance

| target accept | kept standard | rep thr | off-script thr | standard dropped | antidistillation_lam_0.055 dropped | poe_gamma_0.7 dropped |
|---|---|---|---|---|---|---|
| 0.900 | 0.9800 | 0.0401 | 0.0000 | 105 (2.00%) | 525 (10.02%) | 332 (6.34%) |
| 0.950 | 0.9800 | 0.0605 | 0.0000 | 105 (2.00%) | 525 (10.02%) | 332 (6.34%) |
| 0.970 | 0.9843 | 0.0763 | 0.0000 | 82 (1.57%) | 512 (9.77%) | 325 (6.20%) |
| 0.980 | 0.9876 | 0.0853 | 0.0000 | 65 (1.24%) | 504 (9.62%) | 325 (6.20%) |
| 0.990 | 0.9924 | 0.1036 | 0.0000 | 40 (0.76%) | 486 (9.28%) | 323 (6.17%) |
| 0.995 | 0.9952 | 0.1142 | 0.0000 | 25 (0.48%) | 480 (9.16%) | 322 (6.15%) |
| 1.000 | 0.9994 | 0.2393 | 0.0101 | 3 (0.06%) | 465 (8.88%) | 314 (5.99%) |

## Operating point: target acceptance 0.990

| condition | n | dropped | drop rate | loop | off-language | accuracy before | accuracy after |
|---|---|---|---|---|---|---|---|
| standard | 5238 | 40 | 0.76% | 0.76% | 0.00% | 0.8805 | 0.8800 |
| antidistillation_lam_0.055 | 5238 | 486 | 9.28% | 6.03% | 4.56% | 0.8165 | 0.8445 |
| poe_gamma_0.7 | 5238 | 323 | 6.17% | 4.01% | 2.58% | 0.8347 | 0.8600 |

`loop` and `off-language` overlap, so they do not sum to the drop rate. Sampled traces from this operating point are in `trace_degenerate/`.
