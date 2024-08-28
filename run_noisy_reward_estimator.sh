# Initialize GPU index and a counter to manage GPU assignment
GPU_INDEX=0
RUNNING_PER_GPU=3  # Number of processes per GPU
GPUS=${GPUS:-3} # Set this to the number of GPUs available

# Loop through environments and levels
for env in halfcheetah walker2d hopper
do
    for level in medium medium-replay medium-expert
    do
        # Calculate the appropriate GPU to use by cycling through indices
        GPU=$((GPU_INDEX % GPUS))
        
        # Run the command with the selected GPU
        CUDA_VISIBLE_DEVICES=$GPU python run_adv_estimator.py --env=${env}-${level}-v2 \
        --seed 1 --n_step 1 --iter 5 --first_eval_steps 1000000 --bc_eval_steps 1000000 \
        --noise_elem reward --noise_type normal --noise_std 0.1 &
        
        # Update GPU index for the next run
        ((GPU_INDEX++))
        
    done
done

# Wait for any remaining background processes to complete
wait
