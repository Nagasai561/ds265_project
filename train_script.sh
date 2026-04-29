#!/bin/bash
#SBATCH --nodes=1                # Request exactly 1 physical node
#SBATCH --cpus-per-task=12       # Number of CPU cores per task
#SBATCH --gres=gpu:1             # Request 1 GPU (Generic Resource)
#SBATCH --mem=32G                # Total RAM for the node
#SBATCH --job-name=simclr    # Give your job a custom name
#SBATCH --output=result_%j.out   # Standard output file (%j expands to jobId)
#SBATCH --time=04:00:00

set -e
dataset="cifar10"
base="resnet18"

# for step in {201..1..20}; do
# for step in {1..201..20}; do
for step in 5; do
	# step="1,3,6,12,25,50,100"
	# step="100,150,175,188,194,197,199"
	# outfile="./weights/expstep_$step-kiter_20-dataset_$dataset-base_$base.pth"

	outfile="./weights/step_$step-kiter_20-dataset_$dataset-base_$base.pth"
	if [ -f "$outfile" ]; then
		echo "Skipping step $step (already done)"
		continue
	fi

	echo "Doing step $step....."
	source /home/nagasaij/.venv/bin/activate
	python3 run.py --data ../data \
	    --dataset-name "$dataset" \
	    --arch "$base" \
	    -j 12 \
	    --epochs 200 \
	    --batch-size 4096 \
	    --num-clusters 10 \
	    --kmeans-iters 20 \
	    --kmeans-epochs "$step" \
	    --fp16-precision \
	    --file-name "step_$step-kiter_20-dataset_$dataset-base_$base"
done
