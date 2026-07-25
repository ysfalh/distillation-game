# Degeneracy filter report

- Traces: `gsm8k_output_small`
- Tokenizer: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- Conditions: `standard` (n=5238), `antidistillation_lam_0.055` (n=5238), `poe_gamma_0.7` (n=5238)

Two fixed rules, identical for every condition and calibrated on nothing. A trace is dropped if it contains a letter from an unexpected script (run of 1 or more) or if any single token repeats 8 or more times in a row.

## Dropped traces

| condition | n | dropped | drop rate | strange script | repetition | both | accuracy before | accuracy after |
|---|---|---|---|---|---|---|---|---|
| standard | 5238 | 1 | 0.02% | 0.00% | 0.02% | 0.00% | 0.8805 | 0.8805 |
| antidistillation_lam_0.055 | 5238 | 578 | 11.03% | 10.02% | 1.85% | 0.84% | 0.8165 | 0.8348 |
| poe_gamma_0.7 | 5238 | 213 | 4.07% | 3.53% | 0.53% | 0.00% | 0.8347 | 0.8462 |

The two rules overlap, so `strange script` and `repetition` do not sum to the drop rate. Sampled traces are in `trace_degenerate/`.

## Sensitivity to the repetition bar

The strange-script rule is unchanged in every row; only `min_consecutive_copies` moves.

| min consecutive copies | standard dropped | antidistillation_lam_0.055 dropped | poe_gamma_0.7 dropped |
|---|---|---|---|
| 4 | 29 (0.55%) | 611 (11.66%) | 256 (4.89%) |
| 6 | 2 (0.04%) | 580 (11.07%) | 213 (4.07%) |
| 8 | 1 (0.02%) | 578 (11.03%) | 213 (4.07%) |
| 12 | 0 (0.00%) | 575 (10.98%) | 210 (4.01%) |
| 16 | 0 (0.00%) | 573 (10.94%) | 206 (3.93%) |
| 32 | 0 (0.00%) | 571 (10.90%) | 206 (3.93%) |

## Characters that triggered the strange-script rule

| character | name | count |
|---|---|---|
| `水` | CJK UNIFIED IDEOGRAPH-6C34 | 25339 |
| `泉` | CJK UNIFIED IDEOGRAPH-6CC9 | 24599 |
| `的` | CJK UNIFIED IDEOGRAPH-7684 | 4225 |
| `猕` | CJK UNIFIED IDEOGRAPH-7315 | 3551 |
| `苟` | CJK UNIFIED IDEOGRAPH-82DF | 2063 |
| `了` | CJK UNIFIED IDEOGRAPH-4E86 | 1365 |
| `个` | CJK UNIFIED IDEOGRAPH-4E2A | 1226 |
| `桀` | CJK UNIFIED IDEOGRAPH-6840 | 1226 |
| `数` | CJK UNIFIED IDEOGRAPH-6570 | 1098 |
| `刺` | CJK UNIFIED IDEOGRAPH-523A | 1032 |
| `是` | CJK UNIFIED IDEOGRAPH-662F | 1011 |
| `脉` | CJK UNIFIED IDEOGRAPH-8109 | 970 |
| `猴` | CJK UNIFIED IDEOGRAPH-7334 | 918 |
| `量` | CJK UNIFIED IDEOGRAPH-91CF | 917 |
| `激` | CJK UNIFIED IDEOGRAPH-6FC0 | 849 |
