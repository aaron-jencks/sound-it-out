#!/bin/bash
#SBATCH --job-name=p2g_dataset_generation_ipa
#SBATCH --account=PAS2836
#SBATCH --output=/fs/ess/PAS2836/ipa_gpt/jobs/logs/%x-%j.out
#SBATCH --error=/fs/ess/PAS2836/ipa_gpt/jobs/logs/errors/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=cpu-exp

module load miniconda3/24.1.2-py310 cuda/13.2.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sound_it_out_p2g_tools # Activate the branch-specific conda environment

free --giga
export PYTHONPATH=. # Set the python path

# setup paths
scratch_prefix="/fs/scratch/PAS2836/ipa_gpt"
storage_prefix="/fs/ess/PAS2836/ipa_gpt"
datasets_prefix="$storage_prefix/datasets"
checkpoints_prefix="$scratch_prefix/checkpoints"
scratch_datasets_prefix="$scratch_prefix/datasets"
scratch_github_prefix="$scratch_prefix/github"
mkdir -pv $scratch_datasets_prefix $scratch_github_prefix $checkpoints_prefix

cd "$scratch_github_prefix/sound-it-out"
echo "copying local config to dir"
cp "scripts/jobs/cardinal/local_pre.json" "src/transcription/p2g/config/local.json"
if [ $? -ne 0 ]; then
        echo "failed to copy local file"
        exit 1
fi
echo "starting script"
./scripts/build_p2g_dataset.sh ipa
