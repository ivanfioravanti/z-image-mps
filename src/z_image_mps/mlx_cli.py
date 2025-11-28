"""MLX-native CLI for running Z-Image-Turbo on Apple Silicon.

This file focuses on the MLX backend so that users can bypass PyTorch/MPS
entirely.  The API mirrors the torch CLI closely, but the implementation relies
on the `mlx` and `mlx-diffusers` stacks.
"""
import argparse
import os
import secrets
from datetime import datetime
from typing import Tuple

try:  # Import lazily so environments without MLX can still install the package
    import mlx.core as mx
    from mlx_diffusers import DiffusionPipeline  # type: ignore
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "The MLX build requires `mlx` and `mlx-diffusers` on Apple Silicon. "
        "Install with `uv pip install .[mlx]` on macOS or see README for details."
    ) from exc


DEFAULT_PROMPT = (
    "Young Chinese woman in red Hanfu with intricate embroidery, holding a folding fan, "
    "soft outdoor night lighting, cinematic and detailed."
)

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "4:3": (1088, 816),
    "3:4": (816, 1088),
}


def pick_dtype(preferred: str) -> mx.Dtype:
    normalized = preferred.lower()
    if normalized == "float16":
        return mx.float16
    if normalized == "bfloat16":
        return mx.bfloat16
    return mx.float32


def load_mlx_pipeline(dtype: mx.Dtype, attention_backend: str) -> DiffusionPipeline:
    model_id = os.environ.get("MLX_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo-mlx")
    pipe: DiffusionPipeline = DiffusionPipeline.from_pretrained(
        model_id,
        dtype=dtype,
        attn_backend=attention_backend,
    )

    # MLX pipelines are lazy; calling compile() warms up the kernels.
    pipe.compile()
    return pipe


def run_generation(args) -> None:
    dtype = pick_dtype(args.dtype)
    mx.random.seed(args.seed or secrets.randbits(63))

    if args.aspect:
        height, width = ASPECT_RATIOS[args.aspect]
    else:
        height, width = args.height, args.width

    pipe = load_mlx_pipeline(dtype, args.attention_backend)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.expanduser(args.outdir or "output")
    os.makedirs(output_dir, exist_ok=True)

    num_images = max(1, args.num_images)

    for image_index in range(num_images):
        seed = args.seed + image_index if args.seed is not None else secrets.randbits(63)
        mx.random.seed(seed)

        print(
            f"[{image_index + 1}/{num_images}] prompt='{args.prompt}' steps={args.steps} "
            f"guidance={args.guidance_scale} seed={seed} size={width}x{height}"
        )

        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=height,
            width=width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=seed,
        )

        image = result["images"][0]
        suffix = f"-{image_index + 1}" if num_images > 1 else ""

        if args.output:
            user_output = os.path.expanduser(args.output)
            if user_output.endswith(os.sep) or os.path.isdir(user_output):
                os.makedirs(user_output, exist_ok=True)
                filename = os.path.join(user_output, f"z-image-{timestamp}{suffix}.png")
            else:
                base, ext = os.path.splitext(user_output)
                ext = ext or ".png"
                dirpath = os.path.dirname(base) or "."
                os.makedirs(dirpath, exist_ok=True)
                filename = f"{base}{suffix}{ext}" if suffix else f"{base}{ext}"
        else:
            filename = os.path.join(output_dir, f"z-image-{timestamp}{suffix}.png")

        image.save(filename)
        print(f"Saved: {os.path.abspath(filename)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Z-Image-Turbo with Apple MLX (no PyTorch dependency).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-p", "--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt.")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Negative prompt text.")
    parser.add_argument("-s", "--steps", type=int, default=9, help="Number of inference steps.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="CFG guidance scale (Turbo models typically expect 0.0).",
    )
    parser.add_argument("--height", type=int, default=1024, help="Image height (px).")
    parser.add_argument("--width", type=int, default=1024, help="Image width (px).")
    parser.add_argument(
        "--aspect",
        choices=sorted(ASPECT_RATIOS.keys()),
        default=None,
        help="Quick aspect ratio presets; overrides height/width when set.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility.")
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Generate multiple images (seeds increment when a base seed is provided).",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file path.")
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory for outputs (ignored when --output is an explicit file).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="MLX dtype for loading the pipeline.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["sdpa", "flash2", "flash3"],
        default="sdpa",
        help="Attention backend for the DiT transformer (if supported by mlx-diffusers).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
