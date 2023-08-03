TAG=layer_norm REW_SCALE=10.0 TASK=antmaze ALPHA=2.5 GUIDE_COEF=1.0 QF_LAYER_NORM=1 bash launch_job_slurm.sh

TAG=layer_norm REW_SCALE=1.0 REW_BIAS=-1 TASK=antmaze ALPHA=2.5 GUIDE_COEF=1.0 QF_LAYER_NORM=1 bash launch_job_slurm.sh
TAG=layer_norm REW_SCALE=10.0 TASK=antmaze ALPHA=2.5 GUIDE_COEF=1.0 QF_LAYER_NORM=0 bash launch_job_slurm.sh



TAG=online_per_v2 TASK=gym ALPHA=2.5 GUIDE_COEF=1.0 RESAMPLE=True ONLINE_PER=1 PER_TEMP=1.0 bash launch_job_slurm.sh
TAG=online_per_v2 TASK=gym ALPHA=2.5 GUIDE_COEF=1.0 RESAMPLE=True ONLINE_PER=1 PER_TEMP=0.8 bash launch_job_slurm.sh
TAG=online_per_v2 TASK=gym ALPHA=2.5 GUIDE_COEF=1.0 RESAMPLE=True ONLINE_PER=1 PER_TEMP=0.6 bash launch_job_slurm.sh
TAG=online_per_v2 TASK=gym ALPHA=2.5 GUIDE_COEF=1.0 RESAMPLE=True ONLINE_PER=1 PER_TEMP=0.4 bash launch_job_slurm.sh
TAG=online_per_v2 TASK=gym ALPHA=2.5 GUIDE_COEF=1.0 RESAMPLE=True ONLINE_PER=1 PER_TEMP=0.2 bash launch_job_slurm.sh
TAG=online_per_v2 TASK=gym ALPHA=2.5 GUIDE_COEF=1.0 RESAMPLE=True ONLINE_PER=1 PER_TEMP=0 bash launch_job_slurm.sh

TAG=online_per_v2 TASK=gym ALPHA=2.5 GUIDE_COEF=1.0 RESAMPLE=True ONLINE_PER=1 PER_TEMP=0.6 START=2 RUNS=3 bash launch_job_slurm.sh


TAG=similarity TASK=gym ALPHA=2.5 BC_COEF=0 bash launch_job_slurm.sh
TAG=similarity TASK=antmaze ALPHA=2.5 BC_COEF=0 REW_SCALE=10.0 bash launch_job_slurm.sh

TAG=similarity_v2 TASK=gym ALPHA=2.5 BC_COEF=0 bash launch_job_slurm.sh
TAG=similarity_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 bash launch_job_slurm.sh

TAG=similarity_v2_1000 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 bash launch_job_slurm.sh


TAG=percent_v2 START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.1 bash launch_job_slurm.sh
TAG=percent_v2 START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=1 PERCENT=0.1 bash launch_job_slurm.sh
TAG=percent_v2 START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.01 bash launch_job_slurm.sh
TAG=percent_v2 START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=1 PERCENT=0.01 bash launch_job_slurm.sh
TAG=percent_v2 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.5 bash launch_job_slurm.sh
TAG=percent_v2 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=1 PERCENT=0.5 bash launch_job_slurm.sh

TAG=percent_other_norm START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=2 PERCENT=0.1 bash launch_job_slurm.sh
TAG=percent_other_norm START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=3 PERCENT=0.1 bash launch_job_slurm.sh
TAG=percent_other_norm START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=4 LAST_ACT_BOUND=1.0 PERCENT=0.1 bash launch_job_slurm.sh
TAG=percent_other_norm START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=4 LAST_ACT_BOUND=10.0 PERCENT=0.1 bash launch_job_slurm.sh
TAG=percent_other_norm START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=5 LAST_ACT_BOUND=1000.0 PERCENT=0.1 bash launch_job_slurm.sh
TAG=percent_other_norm START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=6 PERCENT=0.1 bash launch_job_slurm.sh # batch norm


TAG=percent_other_norm START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.1 WEIGHT_DECAY=0.01 bash launch_job_slurm.sh
TAG=percent_other_norm START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.1 WEIGHT_DECAY=0.001 bash launch_job_slurm.sh
TAG=percent_other_norm START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.1 WEIGHT_DECAY=0.0001 bash launch_job_slurm.sh

TAG=percent_other_norm START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.1 DROPOUT=0.1 bash launch_job_slurm.sh
TAG=percent_other_norm START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.1 DROPOUT=0.5 bash launch_job_slurm.sh

TAG=percent_v2 START=1 RUNS=2 TASK=gym ALPHA=2.5 BC_COEF=3.0 QF_LAYER_NORM=0 PERCENT=0.1 bash launch_job_slurm.sh
TAG=percent_v2 START=1 RUNS=2 TASK=gym ALPHA=2.5 BC_COEF=10.0 QF_LAYER_NORM=0 PERCENT=0.1 bash launch_job_slurm.sh

TAG=percent_traj START=3 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.1 TRAJ=1 bash launch_job_slurm.sh
TAG=percent_traj START=1 RUNS=2 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=1 PERCENT=0.1 TRAJ=1 bash launch_job_slurm.sh

TAG=percent_traj START=1 RUNS=2 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.01 TRAJ=1 bash launch_job_slurm.sh
TAG=percent_traj START=1 RUNS=2 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=1 PERCENT=0.01 TRAJ=1 bash launch_job_slurm.sh




TAG=percent_eigenvalue START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.01 bash launch_job_slurm.sh
TAG=percent_eigenvalue START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=1 PERCENT=0.01 bash launch_job_slurm.sh


TAG=percent_eigenvalue START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=2 PERCENT=0.01 bash launch_job_slurm.sh
TAG=percent_eigenvalue START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=3 PERCENT=0.01 bash launch_job_slurm.sh
TAG=percent_eigenvalue START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=4 LAST_ACT_BOUND=1.0 PERCENT=0.01 bash launch_job_slurm.sh
TAG=percent_eigenvalue START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=5 LAST_ACT_BOUND=1000.0 PERCENT=0.01 bash launch_job_slurm.sh
TAG=percent_eigenvalue START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=6 PERCENT=0.01 bash launch_job_slurm.sh # batch norm
TAG=percent_eigenvalue START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.01 WEIGHT_DECAY=0.01 bash launch_job_slurm.sh
TAG=percent_eigenvalue START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.01 DROPOUT=0.1 bash launch_job_slurm.sh

TAG=percent_eigenvalue START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=3 PERCENT=0.01 bash launch_job_slurm.sh
TAG=percent_eigenvalue START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=4 LAST_ACT_BOUND=1.0 PERCENT=0.01 bash launch_job_slurm.sh
TAG=percent_eigenvalue START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=5 LAST_ACT_BOUND=1000.0 PERCENT=0.01 bash launch_job_slurm.sh
TAG=percent_eigenvalue START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=6 PERCENT=0.01 bash launch_job_slurm.sh # batch norm
TAG=percent_eigenvalue START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.01 WEIGHT_DECAY=0.01 bash launch_job_slurm.sh
TAG=percent_eigenvalue START=2 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.01 DROPOUT=0.1 bash launch_job_slurm.sh


TAG=regulariztion_eigenvalue START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=20.0 TAU=1.0 QF_LAYER_NORM=0 bash launch_job_slurm.sh
TAG=regulariztion_eigenvalue START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 QF_LAYER_NORM=0 WEIGHT_DECAY=0.01 bash launch_job_slurm.sh
TAG=regulariztion_eigenvalue START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 QF_LAYER_NORM=0 DROPOUT=0.1 bash launch_job_slurm.sh
TAG=regulariztion_eigenvalue START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 QF_LAYER_NORM=1 bash launch_job_slurm.sh

TAG=regulariztion_eigenvalue START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 QF_LAYER_NORM=2 bash launch_job_slurm.sh
TAG=regulariztion_eigenvalue START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 QF_LAYER_NORM=3 bash launch_job_slurm.sh
TAG=regulariztion_eigenvalue START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 QF_LAYER_NORM=5 LAST_ACT_BOUND=1000.0 bash launch_job_slurm.sh
TAG=regulariztion_eigenvalue START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 QF_LAYER_NORM=6 bash launch_job_slurm.sh

# DR3
eval "$(TAG=DR3 START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=1.0 DR3_COEF=0.0001 bash launch_job.sh)"


TAG=DR3 START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=1.0 DR3_COEF=0.0001 bash launch_job_slurm.sh
TAG=DR3 START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=1.0 DR3_COEF=0.001 bash launch_job_slurm.sh
TAG=DR3 START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=1.0 DR3_COEF=0.01 bash launch_job_slurm.sh
TAG=DR3 START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=1.0 DR3_COEF=0.5 bash launch_job_slurm.sh
TAG=DR3 START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=1.0 DR3_COEF=1.0 bash launch_job_slurm.sh
TAG=DR3 START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=1.0 DR3_COEF=10.0 bash launch_job_slurm.sh


TAG=DR3 START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.1 DR3_COEF=0.0001 bash launch_job_slurm.sh
TAG=DR3 START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.1 DR3_COEF=0.001 bash launch_job_slurm.sh
TAG=DR3 START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.1 DR3_COEF=0.01 bash launch_job_slurm.sh
TAG=DR3 START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.1 DR3_COEF=0.5 bash launch_job_slurm.sh
TAG=DR3 START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.1 DR3_COEF=1.0 bash launch_job_slurm.sh
TAG=DR3 START=1 RUNS=1 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.1 DR3_COEF=10.0 bash launch_job_slurm.sh

TAG=OPER-percent START=1 RUNS=2 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.5 BC_EVAL=1 RESAMPLE=True bash launch_job_slurm.sh
TAG=percent_v2 START=2 RUNS=2 TASK=gym ALPHA=2.5 BC_COEF=1.0 QF_LAYER_NORM=0 PERCENT=0.5 bash launch_job_slurm.sh


CUDA_VISIBLE_DEVICES=2 WANDB_API_KEY=3e0863e2d8f819730b85529bd24b3ebbb96d0eb3 python main.py --bc_eval=0 --alpha=2.5 --bc_coef=1.0 --qf_layer_norm=0 --reward_scale=1 --reward_bias=0 --percent=0.1 --traj=0 --last_act_bound=1.0 --weight_decay=0 --dropout_prob=0 --tau=0.005 --dr3_coef=0.0001 --tag=DR3 --seed=1 --env=walker2d-medium-expert-v2 &

