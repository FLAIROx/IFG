#!/bin/bash
# Generate 8 paraphrases per question (temperature 0.6) for the 500-question
# math subset.
python hendrycks_math/paraphrase_questions.py --yaml=hendrycks_math/configs/math_subset/qwen_2.5_7b/paraphrase_config.yaml "$@"
