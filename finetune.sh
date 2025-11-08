train_config_name=$1
model_name=$2
gpu_use=$3
export OPENPI_DATA_HOME="your_OPENPI_DATA_HOME(checkpoints)"
export HF_LEROBOT_HOME="your_HF_LEROBOT_HOME(datasets)"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=$gpu_use
export XDG_CACHE_HOME="your_XDG_CACHE_HOME"
echo $CUDA_VISIBLE_DEVICES
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 python scripts/train.py $train_config_name --exp-name=$model_name --overwrite
