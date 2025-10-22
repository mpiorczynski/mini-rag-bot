#!/bin/bash
#SBATCH --job-name=mini
#SBATCH --account=<FILL_THIS>
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/embedder-%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<FILL_THIS>


set -e
hostname; pwd; date

mkdir -p logs

source .env
eval "$(conda shell.bash hook)"
conda activate mini-rag-dev

cd $PROJECT_DIR/deployment/embedder
uvicorn embedder_api:app --port 8080 --host 0.0.0.0