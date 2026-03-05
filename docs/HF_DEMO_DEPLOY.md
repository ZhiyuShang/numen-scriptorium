# Hugging Face Demo Deployment (Space)

This project now includes a minimal HF Space UI entrypoint: `app.py`.

## Files used by Space

- `app.py`: Gradio demo UI + inference call
- `requirements.txt`: runtime dependencies for Space build

## Environment variables (Space Settings → Variables and secrets)

- `NS_BASE_MODEL` (optional)
  - Default: `Qwen/Qwen2.5-7B-Instruct`
- `NS_ADAPTER` (optional)
  - Default: `outputs/qwen2_5_7b_boh_qlora/best`
  - Can be local path or HF repo id for LoRA adapter
- `NS_USE_4BIT` (optional)
  - `1` = enable 4-bit loading (default, GPU preferred)
  - `0` = normal loading
- `NS_DEFAULT_INSTRUCTION` (optional)
  - Default instruction shown in UI

## Recommended demo setup

1. Create a Gradio Space and push this repository.
2. Use a GPU hardware profile for 7B demo.
3. Ensure `NS_BASE_MODEL` and `NS_ADAPTER` are from the same 7B training setup.
4. If using private model/adapter, set HF token in Space secrets.
