# Results

## Comparison Matrix

| Teacher | Teacher Accuracy | student_naive | student_strategic_fd (beta_s=0.5) |
| --- | --- | --- | --- |
| teacher_antidistillation_lam_0.01 | 0.8718 | 0.5732 | 0.5848<br><sub>mean_a=-0.2045, frac_mass_top20=0.2208</sub> |
| teacher_antidistillation_lam_0.02 | 0.8792 | 0.5840 | 0.5542<br><sub>mean_a=-0.2060, frac_mass_top20=0.2209</sub> |
| teacher_antidistillation_lam_0.03 | 0.7486 | 0.0000 | 0.4069<br><sub>mean_a=-0.2433, frac_mass_top20=0.2242</sub> |
| teacher_antidistillation_lam_0.035 | 0.5095 | 0.0008 | 0.0397<br><sub>mean_a=-0.3407, frac_mass_top20=0.2269</sub> |
| teacher_poe_gamma_0.65 | 0.7800 | 0.4524 | 0.5294<br><sub>mean_a=-0.2119, frac_mass_top20=0.2223</sub> |
| teacher_poe_gamma_0.7 | 0.7932 | 0.2804 | 0.5178<br><sub>mean_a=-0.2061, frac_mass_top20=0.2217</sub> |

Model Context:
- Dataset: `gsm8k`
- Teacher model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- Student model: `meta-llama/Llama-3.2-3B`
