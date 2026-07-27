#!/bin/bash
# Sample one response per paraphrase (output of run_paraphrase_math_subset.sh)
# and report Pass@N once, grouped back by the original question.
# No temperature tuning is done here -- see paraphrase_eval_config.yaml.
python hendrycks_math/paraphrase_infer_and_score.py --yaml=hendrycks_math/configs/math_subset/qwen_2.5_7b/paraphrase_eval_config.yaml "$@"
