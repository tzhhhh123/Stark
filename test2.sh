export CUDA_VISIBLE_DEVICES=0
python tracking/test.py stark_st baseline_mlp --dataset otb --threads 1
python tracking/test.py stark_st baseline_mlp --dataset lasot --threads 1
python tracking/test.py stark_st baseline_mlp --dataset tnl2k --threads 1
