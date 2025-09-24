#!/bin/bash -l

# ================== resnet50
python flop_analy.py \
  --tasks flop \
  --config-file configs/convnext_large_waterovs_id.yaml \
  &> "flop_id_l_s.txt"

python flop_analy.py \
  --tasks parameter \
  --config-file configs/convnext_large_waterovs_id.yaml \
  &> "parameter_id_l_s.txt"