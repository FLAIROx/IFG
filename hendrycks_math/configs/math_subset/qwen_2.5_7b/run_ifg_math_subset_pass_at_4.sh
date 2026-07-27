#!/bin/bash
# Run IFG (multi-step) evaluation on the 500-question math subset,
# reporting Pass@4 (4 samples per question) at temperature_even_index=0.48,
# temperature_odd_index=0.20.
python hendrycks_math/ifg_infer_and_score.py --yaml=hendrycks_math/configs/math_subset/qwen_2.5_7b/ifg_eval_config_pass_at_4.yaml "$@"
