
#!/bin/bash

# Download the SGD evaluation script(s)

# the script(s) require the following
#absl-py>=0.7.0
#fuzzywuzzy>=0.17.0
#numpy>=1.16.1
#six>=1.12.0
#tensorflow==1.14.0

evaluate_url="https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/schema_guided_dst/evaluate.py"
output_file="dsi/eval/sgd/evaluate.py"
curl -L $evaluate_url -o $output_file

metrics_url="https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/schema_guided_dst/metrics.py"
output_file="dsi/eval/sgd/metrics.py"
curl -L $metrics_url -o $output_file