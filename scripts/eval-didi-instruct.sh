module load conda/2024.09
module load gcc/14.1.0
module load cuda/12.6.0
module load jupyter/1.1.1

conda activate mask_model
clear

# checkpoint_path=/input_your_path/ocr-diffusion/didi-instruct/outputs/openwebtext-train/2026.02.02/181645/checkpoints/student_iters_4000.ckpt
# checkpoint_path=/input_your_path/ocr-diffusion/didi-instruct/outputs/openwebtext-train/2026.02.03/182123/checkpoints/student_iters_4800.ckpt
# checkpoint_path=/input_your_path/ocr-diffusion/didi-instruct/outputs/openwebtext-train/2026.02.04/142331/checkpoints/student_iters_2000.ckpt
# checkpoint_path=/input_your_path/ocr-diffusion/didi-instruct/outputs/openwebtext-train/2026.02.04/142331/checkpoints/student_iters_2600.ckpt
# /your_checkpoint_path/didi-instruct.ckpt
ckpt_folder=/input_your_path/ocr-diffusion/didi-instruct/outputs/openwebtext-train/2026.02.04/161449/checkpoints/
clear

steps=16
batch=8
n_batch=2  # total samples = batch * n_batch
echo "----------------------------------------------------------------"
echo "Running evaluation over checkpoints in: $ckpt_folder"
echo "  steps       = $steps"
echo "  min_iter    = 2000"
echo "----------------------------------------------------------------"

# Results file
results_file="${ckpt_folder%/}/eval_results.tsv"
echo -e "checkpoint\titer\tperplexity\tentropy" > "$results_file"

# Iterate over .ckpt files and evaluate those with iter > 2000
for ckpt in "$ckpt_folder"/*.ckpt; do
    [ -f "$ckpt" ] || continue
    base=$(basename "$ckpt")
    # Extract first number group from filename as iter (fallback to 0 if none)
    if [[ "$base" =~ ([0-9]{3,7}) ]]; then
        iter=${BASH_REMATCH[1]}
    else
        iter=0
    fi

    if (( iter > 3000 )); then
        echo "Evaluating $base (iter=$iter) ..."
        checkpoint_path="$ckpt"
        # Capture output to temp log to parse metrics
        out_log=$(mktemp)
        python3 /input_your_path/ocr-diffusion/didi-instruct/main.py \
            mode=sample_eval \
            loader.batch_size="$batch" \
            loader.eval_batch_size="$batch" \
            model=small \
            algo=didi_instruct \
            eval.checkpoint_path="$checkpoint_path" \
            sampling.steps="$steps" \
            sampling.num_sample_batches=$n_batch \
            eval.generate_samples=true \
            sampling.predictor=guided \
            sampling.noise_removal=ancestral 2>&1 | tee "$out_log"

        # Try to extract perplexity and entropy from the output (case-insensitive)
        perplexity=$(grep -Eio 'perplexity[:= ]+[0-9]+(\.[0-9]+)?' "$out_log" | head -1 | grep -Eo '[0-9]+(\.[0-9]+)?' || true)
        # fall back to old 'ppl' token if needed
        if [ -z "$perplexity" ]; then
            perplexity=$(grep -Eoi 'ppl[:= ]+[0-9]+(\.[0-9]+)?' "$out_log" | head -1 | grep -Eo '[0-9]+(\.[0-9]+)?' || true)
        fi
        entropy=$(grep -Eoi 'entropy[:= ]+[0-9]+(\.[0-9]+)?' "$out_log" | head -1 | grep -Eo '[0-9]+(\.[0-9]+)?' || true)

        # Fallbacks if not found
        if [ -z "$perplexity" ]; then perplexity="NA"; fi
        if [ -z "$entropy" ]; then entropy="NA"; fi

        echo -e "${base}\t${iter}\t${perplexity}\t${entropy}" >> "$results_file"
        rm -f "$out_log"
    else
        echo "Skipping $base (iter=${iter})"
    fi
done

echo "\nEvaluation complete. Results written to: $results_file\n"
column -t -s $'\t' "$results_file" || cat "$results_file"
