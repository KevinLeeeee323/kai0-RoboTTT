source .venv/bin/activate
# export WANDB_MODE=${WANDB_MODE:-offline}
RUNNAME=ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD   # or ADVANTAGE_TORCH_PI06_FLATTEN_FOLD
RUNTIME=run1
mkdir -p "./experiment/${RUNNAME}/log"

# Single-GPU
uv run python scripts/train_pytorch.py ${RUNNAME} --exp_name=${RUNTIME} --save_interval 10000 \
    2>&1 | tee "./experiment/${RUNNAME}/log/${RUNTIME}.log"

# # Multi-GPU (e.g. 8 GPUs)
# uv run torchrun --standalone --nproc_per_node=8 scripts/train_pytorch.py ${RUNNAME} \
#     --exp_name=${RUNTIME} --save_interval 10000 \
#     2>&1 | tee "./experiment/${RUNNAME}/log/${RUNTIME}.log"

# # Resume from latest checkpoint
# uv run python scripts/train_pytorch.py ${RUNNAME} --exp_name=${RUNTIME} --resume