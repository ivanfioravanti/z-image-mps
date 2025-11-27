import argparse
import os
import secrets
from datetime import datetime
from typing import Tuple

import torch
from diffusers import ZImagePipeline

from . import __version__


DEFAULT_PROMPT = (
    "Young Chinese woman in red Hanfu with intricate embroidery, holding a folding fan, "
    "soft outdoor night lighting, cinematic and detailed."
)

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1216, 684),
    "9:16": (684, 1216),
    "4:3": (1088, 816),
    "3:4": (816, 1088),
}


def pick_device(preferred: str = "auto") -> Tuple[str, torch.dtype]:
    """Return the best available device and matching dtype."""
    if preferred != "auto":
        normalized = preferred.lower()
        if normalized == "mps" and torch.backends.mps.is_available():
            return "mps", torch.float16
        if normalized == "cuda" and torch.cuda.is_available():
            return "cuda", torch.bfloat16
        if normalized == "cpu":
            return "cpu", torch.float32
        print(f"Warning: Requested device '{preferred}' not available, falling back to auto.")

    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def create_generator(device: str, seed: int) -> torch.Generator:
    # MPS generators must live on CPU
    generator_device = "cpu" if device == "mps" else device
    return torch.Generator(device=generator_device).manual_seed(seed)


def configure_attention(pipe: ZImagePipeline, backend: str) -> None:
    backend_map = {
        "sdpa": None,
        "flash2": "flash",
        "flash3": "_flash_3",
    }
    target = backend_map.get(backend, None)
    if not target:
        return

    try:
        pipe.transformer.set_attention_backend(target)
        print(f"Using attention backend: {backend}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: could not enable {backend} attention ({exc}); using default SDPA.")


def load_pipeline(args, device: str, torch_dtype: torch.dtype) -> ZImagePipeline:
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
    )

    configure_attention(pipe, args.attention_backend)

    if args.cpu_offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if args.compile:
        try:
            pipe.transformer.compile()
            print("Compiled DiT transformer for faster inference (first run will be slower).")
        except Exception as exc:  # pragma: no cover - best-effort optimization
            print(f"Warning: torch.compile failed ({exc}); continuing without compilation.")

    return pipe


def run_generation(args) -> None:
    device, torch_dtype = pick_device(args.device)
    print(f"Using device: {device} (dtype={torch_dtype})")

    pipe = load_pipeline(args, device, torch_dtype)

    if args.aspect:
        height, width = ASPECT_RATIOS[args.aspect]
    else:
        height, width = args.height, args.width
    steps = args.steps
    guidance = args.guidance_scale

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.outdir or "output"
    os.makedirs(output_dir, exist_ok=True)

    num_images = max(1, args.num_images)

    for image_index in range(num_images):
        if args.seed is not None:
            per_image_seed = int(args.seed) + image_index
        else:
            per_image_seed = secrets.randbits(63)

        generator = create_generator(device, per_image_seed)

        print(
            f"[{image_index + 1}/{num_images}] prompt='{args.prompt}' "
            f"steps={steps} guidance={guidance} seed={per_image_seed} size={width}x{height}"
        )

        with torch.inference_mode():
            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )

        image = result.images[0]

        suffix = f"-{image_index + 1}" if num_images > 1 else ""

        if args.output:
            base, ext = os.path.splitext(args.output)
            ext = ext or ".png"
            filename = f"{base}{suffix}{ext}" if suffix else f"{base}{ext}"
        else:
            filename = os.path.join(output_dir, f"z-image-{timestamp}{suffix}.png")
        image.save(filename)
        print(f"Saved: {os.path.abspath(filename)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Z-Image-Turbo locally with a simple CLI (Apple Silicon MPS, CUDA, or CPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"z-image-mps {__version__}",
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
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Force a device or let the CLI pick automatically.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["sdpa", "flash2", "flash3"],
        default="sdpa",
        help="Attention backend for the DiT transformer.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Try torch.compile on the transformer for extra speed (best on CUDA).",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable CPU offload (CUDA only) to reduce VRAM usage.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
