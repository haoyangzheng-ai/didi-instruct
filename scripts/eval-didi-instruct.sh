module load conda/2024.09
module load gcc/14.1.0
module load cuda/12.6.0
module load jupyter/1.1.1

conda activate mask_model

export HF_HOME="/scratch/gautschi/zheng528/model"
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# checkpoint_path=/scratch/gautschi/zheng528/duo/out/mdlm.ckpt  # 1024 38.53; 512 47.79; 256 56.33; 128 80.21; 64 109.90; 32 157.70; 1: 4023.80
# checkpoint_path=/scratch/gautschi/zheng528/duo/out/reinforce-distilled.cpkt
# checkpoint_path=/scratch/gautschi/zheng528/duo/out/reinforce-distilled-twosteps.cpkt

# checkpoint_path=/scratch/gautschi/zheng528/duo/out/diff-distill-1-step.ckpt

# checkpoint_path=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.06/122744/checkpoints/student_iters_200.ckpt

# two steps
# checkpoint_path=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.06/174557/checkpoints/student_iters_8000.ckpt
# checkpoint_path=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.01/160857/checkpoints/student_iters_800.ckpt

# checkpoint_path=/scratch/gautschi/zheng528/duo/out/reinforce-distilled-onestep-ppl-456-entropy-458.ckpt
# checkpoint_path=/scratch/gautschi/zheng528/duo/out/diff_instruct_1step_ppl0698.ckpt

# checkpoint_path=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.08/164119/checkpoints/student_iters_3600.ckpt

# checkpoint_path=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.08/184857/checkpoints/student_iters_3600.ckpt

# checkpoint_path=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.10/105920/checkpoints/student_iters_5200.ckpt

# checkpoint_path=/scratch/gautschi/zheng528/duo/out/mdlm-distill-multi-steps.ckpt
# checkpoint_path=/scratch/gautschi/zheng528/duo/out/mdlm-distill-few-steps.ckpt
# clear

# # # # steps=8
# for steps in 8 # 8 16 32 64 128 # 256 512 1024
# do
#   python3 main.py \
#     mode=sample_eval \
#     loader.batch_size=48 \
#     loader.eval_batch_size=48 \
#     model=small \
#     algo=diff_instruct \
#     algo.T=$steps \
#     eval.checkpoint_path=$checkpoint_path \
#     sampling.steps=$steps \
#     sampling.num_sample_batches=1 \
#     eval.generate_samples=true \
#     +wandb.offline=true \
#     sampling.predictor=ancestral_cache \
#     sampling.noise_removal=ancestral
# done


# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.06/174557/checkpoints/  # remask 0.0
# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.08/072650/checkpoints/  # remask 0.1
# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.07/152210/checkpoints/  # remask 0.2

# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.08/124217/checkpoints/  # lr 5e-6; coupled
# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.08/124233/checkpoints/  # lr 5e-6; decoupled uniform
# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.08/124108/checkpoints/  # lr 5e-6; decoupled bias

# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.08/164119/checkpoints  # lr 1e-6; coupled
# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.08/164209/checkpoints  # lr 1e-6; decoupled bias

# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.08/184812/checkpoints/  # weight mode: constant
# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.08/184857/checkpoints/  # weight mode: complete

# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.08/213520/checkpoints  # Jeffrey
# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.08/213701/checkpoints  # Backward
# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.09/113022/checkpoints

# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.09/144021/checkpoints  # beta25
# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.09/144018/checkpoints  # beta52

# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.09/165907/checkpoints  # beta11
# ckpt_dir=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.09/170007/checkpoints  # beta21

# ckpt_dirs=(
#   # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.09/214916/checkpoints  # beta22 3393064
#   # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.09/220626/checkpoints  # beta52 3393087
#   /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.09/220635/checkpoints  # beta25 3393088
#   /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.10/021136/checkpoints  # beta11 3393464
#   # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.10/021236/checkpoints  # beta55 no test
# )

# # ckpt_dirs=/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.10/105920/checkpoints  # importance, d_lr 1e-6
# # ckpt_dirs=(
# #   # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.10/152338/checkpoints  # importance false, d_lr 5e-6
# #   # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.10/152406/checkpoints  # importance true,  d_lr 5e-6
# #   # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.10/182422/checkpoints  # true, d_lr 5e-6, s_lr 2e-6
# #   /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.10/182505/checkpoints  # false, d_lr 5e-6, s_lr 2e-6
# # )

# ckpt_dirs=(
#   # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.10/021136/checkpoints  # Forward
#   # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.10/085722/checkpoints  # Jeffrey
#   # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.10/085808/checkpoints  # backward
#   # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.10/215007/checkpoints
# #   /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.13/200802/checkpoints/  # 1.0 10.0 0.5 for 200802  wrong sigma
# #   /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.14/080432/checkpoints/
# #   /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.14/080631/checkpoints/
#   # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.14/101819/checkpoints/  # sigma 0.95
# #   /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.14/102030/checkpoints/   # tau 0.95
#   /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.14/163248/checkpoints
# )

ckpt_dir=/scratch/gautschi/zheng528/duo/out/DiPO-8-steps.ckpt

clear

batch=10
n_batch=1
guide_start=1.0 
guide_end=50.0
guide_ratio=0.0

for guide_start in $(seq 0.0 1.0 10.0)
do
  # Middle loop for guide_end, calculated based on guide_start + offset
  for offset in 30.0 10.0 300.0
  do
    # Use 'bc' for floating point arithmetic in shell
    guide_end=$(echo "$guide_start + $offset" | bc)

    # Inner loop for guide_ratio from 0.0 to 1.0
    for guide_ratio in $(seq 0.2 0.6)
    do
      # Your existing loops for steps and checkpoints
      for steps in 8 # You can expand this list again, e.g., 8 16 32
      do
        # for ckpt_dir in "${ckpt_dirs[@]}"
        # do
        #   for ckpt in $ckpt_dir/*.ckpt
        #   do
            echo "----------------------------------------------------------------"
            echo "Running with params:"
            echo "  guide_start = $guide_start"
            echo "  guide_end   = $guide_end"
            echo "  guide_ratio = $guide_ratio"
            echo "  steps       = $steps"
            echo "  checkpoint  = $ckpt_dir"
            echo "----------------------------------------------------------------"
            
            # python3 main.py \
            #     mode=sample_eval \
            #     loader.batch_size=40 \
            #     loader.eval_batch_size=40 \
            #     model=small \
            #     algo=diff_instruct \
            #     algo.T=8 \
            #     eval.checkpoint_path=$ckpt \
            #     sampling.steps=$steps \
            #     sampling.num_sample_batches=1 \
            #     sampling.predictor=ancestral_cache \
            #     sampling.noise_removal=greedy \
            #     eval.generate_samples=true \
            #     +wandb.offline=true
            python3 main.py \
                mode=sample_eval \
                loader.batch_size="$batch" \
                loader.eval_batch_size="$batch" \
                model=small \
                algo=diff_instruct \
                eval.checkpoint_path="$ckpt_dir" \
                sampling.steps="$steps" \
                sampling.num_sample_batches=$n_batch \
                algo.guidance_scale_start=$guide_start \
                algo.guidance_scale_end=$guide_end \
                algo.rerank_steps_ratio=$guide_ratio \
                algo.num_candidates=4 \
                eval.generate_samples=true \
                sampling.predictor=guided \
                sampling.noise_removal=ancestral \
                +wandb.offline=true
        #   done
        # done
      done
    done
  done
done


# # Set your variables
# batch=10
# steps=8
# n_batch=2
# # ckpt="/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.13/180927/checkpoints/student_iters_4000.ckpt"
# ckpt="/scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.13/200802/checkpoints/student_iters_5000.ckpt"


# # python3 main.py \
# #   mode=sample_eval \
# #   loader.batch_size=20 \
# #   loader.eval_batch_size=20 \
# #   model=small \
# #   algo=diff_instruct \
# #   algo.T=8 \
# #   eval.checkpoint_path=$ckpt \
# #   sampling.steps=$steps \
# #   sampling.num_sample_batches=1 \
# #   sampling.predictor=ancestral \
# #   eval.generate_samples=true \
# #   +wandb.offline=true

# # python3 main.py \
# #     mode=sample_eval \
# #     loader.batch_size="$batch" \
# #     loader.eval_batch_size="$batch" \
# #     model=small \
# #     algo=diff_instruct \
# #     eval.checkpoint_path="$ckpt" \
# #     sampling.steps="$steps" \
# #     sampling.num_sample_batches=$n_batch \
# #     sampling.predictor=guided \
# #     algo.guidance_scale_start=0.2 \
# #     algo.guidance_scale_end=1.0 \
# #     algo.rerank_steps_ratio=0.5 \
# #     algo.num_candidates=4 \
# #     eval.generate_samples=true \
# #     sampling.noise_removal=ancestral \
# #     +wandb.offline=true
