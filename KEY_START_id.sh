# !/bin/bash -l

# GPEM:
#     ENABLED: True
#     FUSION: "alpha_fusion" # add | mlp | alpha_fusion
#     DEPTHANYTHING_MODEL: "vitl"  # 'vits' | vitb' | 'vitl'
# SPIM:
#     SELECT_TYPE: "mixed" # mixed | weighted | noselect
#     TOPN: 20 


###### In Domain
# ================== with opencalip convnext base ================== #
python train_net_id.py \
  --num-gpus 4 \
  --config-file configs/convnext_base_waterovs_id.yaml \
  SOLVER.IMS_PER_BATCH 16 \
  MODEL.GPEM.FUSION add \
  MODEL.GPEM.DEPTHANYTHING_MODEL vitb \
  MODEL.SPIM.SELECT_TYPE "mixed" \
  MODEL.SPIM.TOPN 20 \
  OUTPUT_DIR maris_output/0_open_convnext_base_waterovs_id_add_vitb_mixed20

# ================== with opencalip convnext large ================== #
python train_net_id.py \
  --num-gpus 4 \
  --config-file configs/convnext_large_waterovs_id.yaml \
  SOLVER.IMS_PER_BATCH 16 \
  MODEL.GPEM.FUSION alpha_fusion \
  MODEL.GPEM.DEPTHANYTHING_MODEL vitb \
  MODEL.SPIM.SELECT_TYPE "mixed" \
  MODEL.SPIM.TOPN 20 \
  OUTPUT_DIR maris_output/0_open_convnext_large_waterovs_id_alpha_fusion_vitb_mixed20