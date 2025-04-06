import argparse
from pathlib import Path
import torch
import contextlib

from modules.image_captioner import caption_single_image, caption_directory, load_models
from modules.joy_caption_two import *

# Determine the device to use (GPU, MPS, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch_dtype = torch.bfloat16
    autocast = lambda: torch.amp.autocast(device_type="cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    torch_dtype = torch.float32  # MPS doesn't support bfloat16
    autocast = contextlib.nullcontext  # No autocasting on MPS
else:
    device = torch.device("cpu")
    torch_dtype = torch.float32
    autocast = contextlib.nullcontext  # No autocasting on CPU

print(f"Using device: {device}")


def setup_parser():
    parser = argparse.ArgumentParser(description="Image Captioner")

    parser.add_argument(
        "input_image",
        type=str,
        nargs="?",
        help="Path to the image or directory containing images",
    )
    parser.add_argument(
        "--caption_type",
        type=str,
        nargs="+",
        default=["Descriptive"],
        choices=CAPTION_TYPE_MAP.keys(),
        help="Types of captions to generate (multiple values supported)",
    )
    parser.add_argument(
        "--caption_length",
        type=str,
        default="long",
        choices=CAPTION_LENGTH_CHOICES,
        help="Length of the caption",
    )
    parser.add_argument(
        "--extra_options",
        type=int,
        nargs="*",
        default=[11, 14],
        help="Indices of extra options to customize the caption",
    )
    parser.add_argument(
        "--name_input",
        type=str,
        default="",
        help="Person/Character Name (if applicable)",
    )
    parser.add_argument(
        "--custom_prompt",
        type=str,
        default="",
        help="Custom prompt (overrides other settings)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models",
        help="Path to the model checkpoint directory",
    )
    parser.add_argument(
        "--llm_model_id",
        type=str,
        default="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--list_extra_options",
        action="store_true",
        help="List available extra options",
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Overwrite existing caption files",
    )
    parser.add_argument(
        "--append",
        "-a",
        action="store_true",
        help="Append to existing caption files instead of skipping them",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed processing information",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    if args.list_extra_options:
        print("Available extra options:")
        for idx, option in enumerate(EXTRA_OPTIONS_LIST):
            print(f"{idx}: {option}")
        exit(0)

    # Map extra option indices to actual options
    if args.extra_options:
        extra_options = []
        for idx in args.extra_options:
            if 0 <= idx < len(EXTRA_OPTIONS_LIST):
                extra_options.append(EXTRA_OPTIONS_LIST[idx])
            else:
                print(f"Invalid extra option index: {idx}")
                exit(1)
    else:
        extra_options = []

    # test to see if the input_image exists and is a file or directory
    if not args.input_image or not Path(args.input_image).exists():
        print(f"Input image not found: {args.input_image}")
        exit(1)

    input_image_type = "single"
    if args.input_image and Path(args.input_image).is_dir():
        input_image_type = "directory"

    # Load models
    checkpoint_path = Path(args.checkpoint_path)
    (
        clip_model,
        tokenizer,
        text_model,
        image_adapter,
    ) = load_models(
        device, torch_dtype, checkpoint_path, args.llm_model_id, args.verbose
    )

    # Process single image
    if input_image_type == "single":
        caption_single_image(
            device,
            args.input_image,
            args.caption_type,
            args.caption_length,
            extra_options,
            args.name_input,
            args.custom_prompt,
            clip_model,
            tokenizer,
            text_model,
            image_adapter,
            args.append,
            args.overwrite,
            args.verbose,
            autocast,
        )
    # Process images in a directory
    elif input_image_type == "directory":
        caption_directory(
            device,
            args.input_image,
            args.caption_type,
            args.caption_length,
            extra_options,
            args.name_input,
            args.custom_prompt,
            clip_model,
            tokenizer,
            text_model,
            image_adapter,
            args.append,
            args.overwrite,
            args.verbose,
            autocast,
        )
    else:
        print(f"Invalid input image type: {input_image_type}")
        exit(1)
