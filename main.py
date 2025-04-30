from datetime import timedelta
import os
import argparse
from pathlib import Path
import time
from typing import List
import torch
import contextlib
from PIL import Image
import re
import hashlib

from modules.joy_two_captioner import (
    caption_image as joy_caption_image,
    load_models as load_joy_models,
)
from modules.joy_caption_two import *

from modules.florence_two_captioner import (
    caption_images as florence_caption_images,
    load_models as load_florence_models,
)
from modules.florence_two import *

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

# Helper function to calculate file hash
def calculate_file_hash(filepath: Path, hash_algo="sha256", buffer_size=65536) -> str:
    """Calculates the hash of a file."""
    hasher = hashlib.new(hash_algo)
    try:
        with open(filepath, 'rb') as f:
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                hasher.update(data)
        return hasher.hexdigest()
    except IOError as e:
        print(f"Warning: Could not read file {filepath} for hashing: {e}")
        return ""
    except Exception as e:
        print(f"Warning: Error hashing file {filepath}: {e}")
        return ""

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
        choices=list(CAPTION_TYPE_MAP.keys()) + list(TASK_TYPE_MAP.keys()),
        help="Types of captions to generate (multiple values supported)",
    )
    parser.add_argument(
        "--caption_length",
        type=str,
        default="medium-length",
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
        default="unsloth/Meta-Llama-3.1-8B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--florence2_model_id",
        type=str,
        default="MiaoshouAI/Florence-2-large-PromptGen-v2.0",
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
    parser.add_argument(
        "--tags",
        "-t",
        type=str,
        nargs="+",
        default=[],
        help="Static tags to prepend to the output file, comma-separated.",
    )

    return parser


def process_images(
    input_path: Path,
    images: List[Path],
    args: argparse.Namespace,
    mapped_extra_options: List[str],
    florence2_model,
    florence2_processor,
    clip_model,
    tokenizer,
    text_model,
    image_adapter,
):
    processed_count = 0
    skipped_count = 0
    appended_count = 0
    error_count = 0
    total_count = len(images)

    # Time tracking variables
    start_time = time.time()
    processed_time = 0

    # Function to format time remaining
    def format_time(seconds):
        if seconds < 0:
            return "Unknown"
        return str(timedelta(seconds=int(seconds)))

    for idx, image_path in enumerate(images, 1):
        file_start_time = time.time()

        # Create output file path with same name but .txt extension
        output_file_path = image_path.with_suffix(".txt")

        # Check if output file exists
        file_exists = output_file_path.exists()

        # Determine file mode based on options
        action = ""
        file_mode = "w"  # Default is write (create or overwrite)
        if file_exists:
            if args.append:
                file_mode = "a"  # Append to existing file
                action = "Appending existing caption for"
                appended_count += 1
            elif args.overwrite:
                action = "Overwriting caption for"
                processed_count += 1
            else:
                # Print skipped files on a new line to maintain history
                print(f"Skipped: {image_path.name} (file already exists)")
                skipped_count += 1

                # Calculate and display time estimates
                elapsed = time.time() - start_time
                if processed_count + appended_count > 0:
                    avg_time = processed_time / (processed_count + appended_count)
                    remaining_files = total_count - idx
                    est_time = avg_time * remaining_files

                    # Clear line and show progress with time estimate
                    print(" " * 150, end="\r")
                    print(
                        f"Progress: {idx}/{total_count} [{processed_count} processed, {appended_count} appended, {skipped_count} skipped, {error_count} errors] - Elapsed: {format_time(elapsed)} - Remaining: {format_time(est_time)}",
                        end="\r",
                    )

                continue
        else:
            action = "Processing"
            processed_count += 1

        # Calculate time estimates
        elapsed = time.time() - start_time
        est_time = "Calculating..."
        if processed_count + appended_count > 0:
            avg_time = processed_time / (processed_count + appended_count)
            remaining_files = (
                total_count - idx + 1
            )  # +1 because we're including the current file
            est_time = format_time(avg_time * remaining_files)

        # Show which file is currently being processed (with carriage return)
        if input_path.is_dir():
            relative_path = image_path.relative_to(input_path)
        else:
            relative_path = image_path
        status = f"Working: {action} {relative_path} ({idx}/{total_count}) - Elapsed: {format_time(elapsed)} - Remaining: {est_time}"
        padding = " " * 50  # Much more padding to ensure the line is cleared
        print(f"{status}{padding}", end="\r")

        try:
            input_image = Image.open(image_path)

            # Convert RGBA images to RGB to avoid channel mismatch
            if input_image.mode == "RGBA":
                input_image = input_image.convert("RGB")

            # Process each caption type
            with open(output_file_path, file_mode, encoding="utf-8") as output_file:
                # Write static tags first if in write/overwrite mode and tags are provided
                wrote_static_tags = False
                if file_mode == "w" and args.tags:
                    output_file.write(", ".join(args.tags))
                    wrote_static_tags = True

                first_caption = True
                # Define task types that should be output as comma-separated tags
                tag_tasks = {
                    "Booru tag list",
                    "Booru-like tag list",
                    "prompt_gen_tags", # Florence-2 tag generation
                }

                for task in args.caption_type:
                    generated_output = None
                    if task in TASK_TYPE_MAP.keys():
                        (
                            _out_tensor,
                            _out_mask_tensor,
                            out_results,
                            _out_data,
                        ) = florence_caption_images(
                            device,
                            torch_dtype,
                            [input_image],
                            args.custom_prompt,
                            task,
                            florence2_model,
                            florence2_processor,
                            verbose=args.verbose,
                        )
                        generated_output = out_results
                    elif task in CAPTION_TYPE_MAP.keys():
                        prompt_used, caption_out = joy_caption_image(
                            device,
                            input_image,
                            task,
                            args.caption_length,
                            mapped_extra_options,
                            args.name_input,
                            args.custom_prompt,
                            clip_model,
                            tokenizer,
                            text_model,
                            image_adapter,
                            args.verbose,
                            autocast,
                        )
                        generated_output = caption_out # Only use the caption part
                    else:
                        print(f"Invalid caption type: {task}")
                        exit(1)

                    # Post-process if it's a tag task
                    if task in tag_tasks and generated_output:
                        # Remove potential introductory lines (like "Here's a list...")
                        lines = generated_output.strip().split('\n')
                        # Heuristic: find the first line that likely contains tags (e.g., starts with '*' or '-') or just take all non-empty lines
                        tag_lines = [line for line in lines if line.strip()]
                        if tag_lines and any(phrase in tag_lines[0] for phrase in ["Here's a list", "Tags:", "list of Booru tags"]):
                             tag_lines = tag_lines[1:] # Skip potential intro line

                        cleaned_tags = []
                        for line in tag_lines:
                            # Remove leading/trailing whitespace and bullet points/hyphens
                            cleaned_line = line.strip().lstrip('*- ').strip()
                            if cleaned_line:
                                # Split potentially comma-separated tags already on one line
                                cleaned_tags.extend([tag.strip() for tag in cleaned_line.split(',') if tag.strip()])
                        generated_output = ", ".join(cleaned_tags)

                    # Write the generated output (potentially cleaned)
                    if generated_output:
                        prefix = ""
                        if file_mode == "a":
                            # Always add newlines when appending
                            prefix = f"{os.linesep}{os.linesep}"
                        elif first_caption:
                            # Add separator only if static tags were written
                            if wrote_static_tags:
                                prefix = ", "
                        else:
                            # Add newlines between multiple generated captions in write mode
                            prefix = f"{os.linesep}{os.linesep}"

                        output_file.write(f"{prefix}{generated_output}")
                        first_caption = False

            # Calculate how long this file took
            file_time = time.time() - file_start_time
            processed_time += file_time

            # Then print completed file on a new line to maintain history
            message = f"Completed: {action} {image_path.name} in {format_time(file_time)}"
            print(f"{message}{' ' * (os.get_terminal_size().columns - len(message))}")

        except Exception as e:
            error_count += 1
            message = f"Error: {image_path.name} - {str(e)}"
            print(f"{message}{' ' * (os.get_terminal_size().columns - len(message))}")

            # if there's only one image, raise the error
            if len(images) == 1:
                raise e
            else:
                continue

    # Calculate total elapsed time
    total_elapsed = time.time() - start_time

    # Print summary at the end
    if input_path.is_dir():
        print(f"\nFinished processing images in {input_path}")
    else:
        print(f"\nFinished processing image {input_path}")
    print(
        f"Processed: {processed_count}, Appended: {appended_count}, Skipped: {skipped_count}, Errors: {error_count}, Total: {total_count}"
    )
    print(f"Total time: {format_time(total_elapsed)}")


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    if args.list_extra_options:
        print("Available extra options:")
        for idx, option in enumerate(EXTRA_OPTIONS_LIST):
            print(f"{idx}: {option}")
        exit(0)

    # Map extra option indices to actual options
    mapped_extra_options = []
    if args.extra_options:
        for idx in args.extra_options:
            if 0 <= idx < len(EXTRA_OPTIONS_LIST):
                mapped_extra_options.append(EXTRA_OPTIONS_LIST[idx])
            else:
                print(f"Invalid extra option index: {idx}")
                exit(1)

    # test to see if the input_image exists and is a file or directory
    if not args.input_image or not Path(args.input_image).exists():
        print(f"Input image not found: {args.input_image}")
        exit(1)

    input_image_type = "single"
    if args.input_image and Path(args.input_image).is_dir():
        input_image_type = "directory"

    # Load models
    checkpoint_path = Path(args.checkpoint_path)

    clip_model = None
    tokenizer = None
    text_model = None
    image_adapter = None
    florence2_model = None
    florence2_processor = None

    if any(task in CAPTION_TYPE_MAP.keys() for task in args.caption_type):
        (
            clip_model,
            tokenizer,
            text_model,
            image_adapter,
        ) = load_joy_models(
            device, torch_dtype, checkpoint_path, args.llm_model_id, args.verbose
        )

    if any(task in TASK_TYPE_MAP.keys() for task in args.caption_type):
        (florence2_model, florence2_processor) = load_florence_models(
            device, torch_dtype, checkpoint_path, args.florence2_model_id, args.verbose
        )

    # Process single image
    if input_image_type == "single":
        input_path = Path(args.input_image)
        if not input_path.is_file():
            print(f"Input image not found: {input_path}")
            exit(1)

        output_file_path = input_path.with_suffix(".txt")
        image_files = [input_path]
        process_images(
            input_path,
            image_files,
            args,
            mapped_extra_options,
            florence2_model,
            florence2_processor,
            clip_model,
            tokenizer,
            text_model,
            image_adapter,
        )

    # Process images in a directory
    elif input_image_type == "directory":
        input_path = Path(args.input_image)
        if not input_path.is_dir():
            print(f"Input directory not found: {input_path}")
            exit(1)

        # List of image file extensions
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]

        # Collect all image files in the directory and subdirectories recursively
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"**/*{ext}"))
            image_files.extend(
                input_path.glob(f"**/*{ext.upper()}")
            )  # Also include uppercase extensions

        # --- Deduplicate the list of found files ---
        image_files = sorted(list(set(image_files)))
        # --- End deduplication ---

        if not image_files:
            print(
                f"No image files found in directory: {input_path} (including subdirectories)"
            )
            exit(1)

        print(f"Found {len(image_files)} images in {input_path} and its subdirectories")

        # --- Check for duplicate images based on hash ---
        print("Checking for duplicate images (this may take a moment)...")
        image_hashes = {}
        for img_path in image_files:
            file_hash = calculate_file_hash(img_path)
            if file_hash:
                if file_hash not in image_hashes:
                    image_hashes[file_hash] = []
                image_hashes[file_hash].append(img_path)

        duplicates_found = False
        for file_hash, paths in image_hashes.items():
            if len(paths) > 1:
                if not duplicates_found:
                    print("\n--- Duplicate Image Warnings ---")
                    duplicates_found = True
                print(f"Warning: The following images appear to be identical (Hash: {file_hash[:8]}...):")
                for p in paths:
                    try:
                        # Attempt to show relative path if possible
                        relative_p = p.relative_to(input_path)
                        print(f"  - {relative_p}")
                    except ValueError:
                        # Fallback to absolute path if not under input_path
                        print(f"  - {p}")
        if duplicates_found:
            print("--- End Duplicate Warnings ---\n")
        else:
             print("No duplicate images found.")
        # --- End duplicate check ---

        process_images(
            input_path,
            image_files,
            args,
            mapped_extra_options,
            florence2_model,
            florence2_processor,
            clip_model,
            tokenizer,
            text_model,
            image_adapter,
        )
    else:
        print(f"Invalid input image type: {input_image_type}")
        exit(1)
