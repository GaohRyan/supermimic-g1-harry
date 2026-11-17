# source tools/offline/env.sh   (from the supermimic root; run tools/offline/setup_offline_env.sh once first)
export SM_ROOT="$(pwd)"
export SM_VARIANT="$SM_ROOT/variants/cuboi-g1/corrected-baseline"
export PYTHONPATH="$SM_VARIANT/src:$SM_ROOT/tools/offline/.mjlab_stub"
export G1_XML="$SM_ROOT/upstream/references/mjlab-8bb2bc9/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
export G1_CONSTANTS="$SM_ROOT/upstream/references/mjlab-8bb2bc9/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py"
export NPZ40_LOCAL="$SM_ROOT/data/processed/cuboi-g1/lafan40-local"
