# Z-Image MPS

Generate images locally with **Tongyi-MAI/Z-Image-Turbo** using a tiny CLI that works on Apple Silicon (MPS), CUDA, or CPU. The project mirrors the `qwen-image-mps` workflow but uses the new Z-Image Diffusers pipeline.

## Highlights
- Auto device pick: prefers MPS, then CUDA, else CPU
- Sensible defaults for Z-Image-Turbo (9 steps, CFG 0.0)
- Aspect presets plus manual height/width overrides
- Optional `torch.compile`, FlashAttention 2/3 switches, and CPU offload (CUDA)
- `uv`-first: run without installing, or install/edit via `uv pip install -e .`

## Quick start

1) Install Python 3.10+ and ensure you have PyTorch with the right backend (MPS or CUDA).

2) Diffusers needs Z-Image support. The dependency is already pointed at the latest diffusers `main` in `pyproject.toml` so `uv` will fetch it automatically (no extra flag needed).

3) Run with `uv` (no global install):
```bash
uv run z-image-mps.py --help
uv run z-image-mps.py -p "A cozy neon-lit alley, cinematic, raining softly" --aspect 16:9
```

Or install locally in editable mode:
```bash
uv pip install -e .
z-image-mps --prompt "Sunlit living room, mid-century modern, natural light"
```

Images are saved to `output/` by default with timestamped filenames.

## Gradio demo

Launch a simple UI (generate-only, no LoRA/edit):
```bash
uv run z-image-mps-gradio --host 0.0.0.0 --port 7860
# or
uv run python -m z_image_mps.gradio_app
```

The UI exposes prompt, negative prompt, steps, guidance (defaults to 0.0), aspect/custom size, seed, device selection, attention backend (SDPA/Flash2/Flash3), optional `torch.compile`, and CUDA CPU-offload.

## CLI reference

```
z-image-mps --prompt "..." [options]

-p, --prompt            Text prompt (default: Hanfu prompt from the Z-Image README)
--negative-prompt       Negative prompt text
-s, --steps             Inference steps (default: 9)
--guidance-scale        CFG scale (Turbo expects 0.0)
--aspect {1:1,16:9,9:16,4:3,3:4}  (optional; uses height/width when omitted)
--height/--width        Exact dimensions (default 1024x1024 when no aspect is set)
--seed                  Seed (incremented per image when generating multiples)
--num-images            Number of images to generate
-o, --output            Output file (otherwise saved to output/)
--outdir                Output directory
--device {auto,mps,cuda,cpu}
--attention-backend     sdpa | flash2 | flash3
--compile               Try torch.compile() on the DiT transformer
--cpu-offload           Enable CPU offload (CUDA only)
```

Notes:
- Guidance should stay at `0.0` for the Turbo checkpoint.
- FlashAttention requires compatible hardware/drivers; the CLI falls back to SDPA if it fails.
- `torch.compile` speeds up repeated runs but makes the first call slower.

## Examples

```bash
# Square default
z-image-mps -p "Analog film portrait of a skateboarder, shallow depth of field"

# Widescreen
z-image-mps -p "Cyberpunk night market, neon haze" --aspect 16:9

# Multiple images with a fixed seed (increments per image)
z-image-mps -p "Nordic fjord at dawn, misty" --num-images 3 --seed 123

# FlashAttention 2 and compiled transformer (CUDA)
z-image-mps -p "A futuristic tram in the rain" --attention-backend flash2 --compile
```

## Why “MPS”?

The original `qwen-image-mps` project focused on making Apple Silicon a first-class citizen. This repo keeps the same spirit: MPS when available, CUDA when present, CPU as a fallback. Everything is packaged to work smoothly with `uv` so you can try Z-Image quickly on a MacBook or GPU box.
