source .venv/bin/activate
export WANDB_MODE=disabled   # ${WANDB_MODE:-offline}

TRANSFORMERS_DIR=$(uv run python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
echo "Transformers dir: $TRANSFORMERS_DIR"
cp -r ./src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_DIR"/

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