# Degeneracy filter report

- Traces: `gsm8k_output_small_mismatch`
- Tokenizer: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- Conditions: `antidistillation_lam_0.03` (n=5238), `antidistillation_lam_0.035` (n=5238), `poe_gamma_0.65` (n=5238), `poe_gamma_0.7` (n=5238)

Two fixed rules, identical for every condition and calibrated on nothing. A trace is dropped if it contains a letter from an unexpected script (run of 1 or more) or if any single token repeats 8 or more times in a row.

## Dropped traces

| condition | n | dropped | drop rate | strange script | repetition | both | accuracy before | accuracy after |
|---|---|---|---|---|---|---|---|---|
| antidistillation_lam_0.03 | 5238 | 581 | 11.09% | 5.46% | 6.57% | 0.94% | 0.7541 | 0.7951 |
| antidistillation_lam_0.035 | 5238 | 2369 | 45.23% | 21.99% | 33.81% | 10.58% | 0.5326 | 0.6755 |
| poe_gamma_0.65 | 5238 | 14 | 0.27% | 0.04% | 0.23% | 0.00% | 0.8436 | 0.8440 |
| poe_gamma_0.7 | 5238 | 9 | 0.17% | 0.10% | 0.08% | 0.00% | 0.8303 | 0.8309 |

The two rules overlap, so `strange script` and `repetition` do not sum to the drop rate. Sampled traces are in `trace_degenerate/`.

## Sensitivity to the repetition bar

The strange-script rule is unchanged in every row; only `min_consecutive_copies` moves.

| min consecutive copies | antidistillation_lam_0.03 dropped | antidistillation_lam_0.035 dropped | poe_gamma_0.65 dropped | poe_gamma_0.7 dropped |
|---|---|---|---|---|
| 4 | 619 (11.82%) | 2397 (45.76%) | 72 (1.37%) | 60 (1.15%) |
| 6 | 583 (11.13%) | 2374 (45.32%) | 16 (0.31%) | 9 (0.17%) |
| 8 | 581 (11.09%) | 2369 (45.23%) | 14 (0.27%) | 9 (0.17%) |
| 12 | 579 (11.05%) | 2360 (45.06%) | 14 (0.27%) | 9 (0.17%) |
| 16 | 571 (10.90%) | 2263 (43.20%) | 11 (0.21%) | 9 (0.17%) |
| 32 | 495 (9.45%) | 2003 (38.24%) | 11 (0.21%) | 9 (0.17%) |

## Characters that triggered the strange-script rule

| character | name | count |
|---|---|---|
| `子` | CJK UNIFIED IDEOGRAPH-5B50 | 93267 |
| `鼻` | CJK UNIFIED IDEOGRAPH-9F3B | 92975 |
| `舞` | CJK UNIFIED IDEOGRAPH-821E | 32520 |
| `蹈` | CJK UNIFIED IDEOGRAPH-8E48 | 30062 |
| `郈` | CJK UNIFIED IDEOGRAPH-90C8 | 29419 |
| `互` | CJK UNIFIED IDEOGRAPH-4E92 | 12106 |
| `的` | CJK UNIFIED IDEOGRAPH-7684 | 11760 |
| `媱` | CJK UNIFIED IDEOGRAPH-5AB1 | 8937 |
| `菇` | CJK UNIFIED IDEOGRAPH-83C7 | 8129 |
| `蘑` | CJK UNIFIED IDEOGRAPH-8611 | 8124 |
| `惠` | CJK UNIFIED IDEOGRAPH-60E0 | 7498 |
| `敏` | CJK UNIFIED IDEOGRAPH-654F | 6788 |
| `灵` | CJK UNIFIED IDEOGRAPH-7075 | 5892 |
| `捯` | CJK UNIFIED IDEOGRAPH-636F | 5434 |
| `是` | CJK UNIFIED IDEOGRAPH-662F | 4959 |
