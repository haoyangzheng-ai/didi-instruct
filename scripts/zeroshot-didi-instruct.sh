module load conda/2024.09
module load gcc/14.1.0
module load cuda/12.6.0
module load jupyter/1.1.1

conda activate mask_model

export HF_HOME="/scratch/gautschi/zheng528/model"
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# checkpoint_path=/scratch/gautschi/zheng528/duo/out/mdlm-distill-few-steps.ckpt
checkpoint_path=/scratch/gautschi/zheng528/duo/out/mdlm.ckpt
checkpoint_path=/scratch/gautschi/zheng528/duo/out/duo-distilled.ckpt
checkpoint_path=/scratch/gautschi/zheng528/duo/out/duo.ckpt

datasets=(
          "ptb"
          "ag_news"
          "lambada"
          "wikitext2"
          "wikitext103"
          "lm1b-gpt2"
          "scientific_papers_arxiv"
          "scientific_papers_pubmed"
          # "openwebtext-split"
          )

ckpt_dirs=(
  # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.10/021136/checkpoints  # Forward
  # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.10/085722/checkpoints  # Jeffrey
  # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.10/085808/checkpoints  # backward
  # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.10/215007/checkpoints
#   /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.13/200802/checkpoints/  # 1.0 10.0 0.5 for 200802  wrong sigma
#   /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.14/080432/checkpoints/
#   /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.14/080631/checkpoints/
  /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.14/101819/checkpoints/  # sigma 0.95
#   /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.14/102030/checkpoints/   # tau 0.95
  # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.14/163248/checkpoints  # lr 2e-6 4 hours
  # /scratch/gautschi/zheng528/duo/outputs/openwebtext-train/2025.09.15/184647/checkpoints
)

clear

batch=16
n_batch=1
gpus=1
for data in "${datasets[@]}"; do
  for steps in 8 # You can expand this list again, e.g., 8 16 32
  do
    # for ckpt_dir in "${ckpt_dirs[@]}"
    # do
    #   for ckpt in $ckpt_dir/*.ckpt
    #   do
        echo "----------------------------------------------------------------"
        echo "Running with params:"
        echo "  data        = $data"
        echo "  steps       = $steps"
        echo "  checkpoint  = $checkpoint_path"
        echo "----------------------------------------------------------------"
        # srun --ntasks-per-node=$gpus --ntasks-per-node=$gpus \
        # python -u -m main \
        python3 main.py \
            mode=ppl_eval \
            loader.batch_size=16 \
            loader.eval_batch_size=16 \
            loader.eval_global_batch_size=128 \
            data="$data" \
            data.insert_valid_eos=False \
            sampling.steps="$steps" \
            sampling.num_sample_batches=$n_batch \
            model=small \
            algo=duo \
            model.length=1024 \
            eval.checkpoint_path=$checkpoint_path \
            +wandb.offline=true
            # data.valid_ratio=0.001 \
    #   done
    # done
  done
done


# for data in "${datasets[@]}"; do
#   echo "$data"
#   srun python -u -m main \
#     mode=ppl_eval \
#     loader.batch_size=16 \
#     loader.eval_batch_size=16 \
#     loader.eval_global_batch_size=128 \
#     data="$data" \
#     data.insert_valid_eos=False \
#     sampling.steps="$steps" \
#     model=small \
#     algo=mdlm \
#     model.length=1024 \
#     eval.checkpoint_path=$checkpoint_path \
#     +wandb.offline=true
# done

#!/bin/bash
#SBATCH -J zeroshot_mdlm_noeos              # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

