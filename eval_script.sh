#!/bin/bash
#SBATCH --nodes=1                # Request exactly 1 physical node
#SBATCH --cpus-per-task=12       # Number of CPU cores per task
#SBATCH --gres=gpu:1             # Request 1 GPU (Generic Resource)
#SBATCH --mem=32G                # Total RAM for the node
#SBATCH --job-name=simclr    # Give your job a custom name
#SBATCH --output=result_%j.out   # Standard output file (%j expands to jobId)
#SBATCH --time=04:00:00

set -e
mkdir -p ./eval_metrics
dataset="cifar10"
base="resnet18"


source /home/nagasaij/.venv/bin/activate

for weight_file in ./weights/*.pth; do
	eval_metrics_file="./eval_metrics/$(basename $weight_file .pth)"
	if [ -f "$eval_metrics_file.csv" ]; then
		echo -e "For weight file $weight_file, corresponding $eval_metrics_file exists"
		echo "So, skipping it"
		continue
	fi

	python3 classifier.py --data ../data \
	    --dataset-name "$dataset" \
	    --arch "$base" \
	    -j 12 \
	    --epochs 100 \
	    --batch-size 4096 \
	    --fp16-precision \
	    --weights-path "$weight_file" \
	    --file-name "$eval_metrics_file.csv"
done
