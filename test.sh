export CUDA_VISIBLE_DEVICES=1,2,3
python tracking/test.py stark_st baseline_mlp_fuse_v2 --dataset tnl2k --threads 3
#python tracking/test.py stark_st baseline_mlp --dataset otb --threads 3
#python tracking/test.py stark_st baseline_mlp --dataset tnl2k --threads 3
