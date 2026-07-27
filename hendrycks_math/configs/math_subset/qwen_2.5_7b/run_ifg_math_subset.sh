#!/bin/bash
# Run IFG (multi-step) evaluation on the 500-question math subset.
# No temperature tuning is done here -- temperature_even_index and
# temperature_odd_index must be supplied on the command line, e.g.:
#   ./run_ifg_math_subset.sh --temperature_even_index=0.6 --temperature_odd_index=0.3
python hendrycks_math/ifg_infer_and_score.py --yaml=hendrycks_math/configs/math_subset/qwen_2.5_7b/ifg_eval_config.yaml "$@"
