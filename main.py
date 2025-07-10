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
import yaml

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

from modules.openai_captioner import (
    caption_image_with_openai,
    load_openai_models,
)

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

def load_yaml_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config or {}
    except FileNotFoundError:
        print(f"Warning: Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config file {config_path}: {e}")
        return {}

def create_sample_config(output_path: str = "config.yaml"):
    """Create a sample YAML configuration file."""
    sample_config = {
        "caption_type": ["Descriptive"],
        "caption_length": "medium-length",
        "extra_options": [11, 14],
        "name_input": "",
        "custom_prompt": "",
        "checkpoint_path": "models",
        "llm_model_id": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "florence2_model_id": "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
        "overwrite": False,
        "append": False,
        "verbose": False,
        "tags": [],
        "flat": False,
        "openai_endpoint": "",
        "openai_api_key": "",
        "openai_model": "gpt-4o",
        "openai_prompt": "Write a medium-length list of Booru-like tags for this image.",
        "include_image_name": False,
        "include_image_path": 0,
        "output_path": "./output"
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f, default_flow_style=False, sort_keys=False)
        print(f"Sample configuration file created: {output_path}")
    except Exception as e:
        print(f"Error creating sample config file: {e}")

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
        choices=list(CAPTION_TYPE_MAP.keys()) + list(TASK_TYPE_MAP.keys()) + ["openai"],
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
    parser.add_argument(
        "--flat",
        "-f",
        action="store_true",
        help="Output processed images and captions to ./output with sequential names",
    )
    parser.add_argument(
        "--openai_endpoint",
        type=str,
        default="",
        help="OpenAI-compatible API endpoint URL (required for 'openai' caption type)",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default="",
        help="API key for OpenAI-compatible endpoint (required for 'openai' caption type)",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-4o",
        help="Model name for OpenAI-compatible API (default: gpt-4o)",
    )
    parser.add_argument(
        "--openai_prompt",
        type=str,
        default="Write a medium-length list of Booru-like tags for this image.",
        help="Prompt to send with image to OpenAI-compatible API",
    )
    parser.add_argument(
        "--include_image_name",
        action="store_true",
        help="True if the image name should be included in the output",
    )
    parser.add_argument(
        "--include_image_path",
        type=int,
        default=0,
        help="Include the specified number of segments of the image path in the output",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="Output directory for caption files (default: ./output)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--create_config",
        type=str,
        metavar="FILENAME",
        help="Create a sample YAML configuration file",
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

    # Create output directory
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Time tracking variables
    start_time = time.time()
    processed_time = 0

    # Function to format time remaining
    def format_time(seconds):
        if seconds < 0:
            return "Unknown"
        return str(timedelta(seconds=int(seconds)))

    # Function to create progress bar
    def create_progress_bar(current, total, width=50):
        progress = current / total if total > 0 else 0
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        percentage = progress * 100
        return f"[{bar}] {percentage:.1f}%"

    # Print initial status
    print(f"Starting to process {total_count} image(s)...")
    print(f"Caption types: {', '.join(args.caption_type)}")
    print(f"Output directory: {args.output_path}")
    if args.flat:
        print(f"Output mode: Flat (sequential naming)")
    print("-" * 80)

    for idx, image_path in enumerate(images, 1):
        file_start_time = time.time()

        # Create output file path
        if args.flat:
            # Use sequential numbering for flat output
            seq_num = f"{idx:03d}"
            output_file_path = output_dir / f"{seq_num}.txt"
            output_image_path = output_dir / f"{seq_num}{image_path.suffix}"
        else:
            # Use output directory while preserving relative path structure
            if input_path.is_dir():
                # For directory input, preserve relative path structure
                relative_path = image_path.relative_to(input_path)
                output_file_path = output_dir / relative_path.with_suffix(".txt")
                output_image_path = image_path  # Keep original image path
                # Ensure the output directory structure exists
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # For single file input, just use the filename
                output_file_path = output_dir / f"{image_path.stem}.txt"
                output_image_path = image_path  # Keep original image path

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
                print(f"⏭️  Skipped: {image_path.name} (file already exists)")
                skipped_count += 1

                # Calculate and display time estimates
                elapsed = time.time() - start_time
                if processed_count + appended_count > 0:
                    avg_time = processed_time / (processed_count + appended_count)
                    remaining_files = total_count - idx
                    est_time = avg_time * remaining_files

                    # Show progress bar and status
                    progress_bar = create_progress_bar(idx, total_count)
                    print(f"\r{progress_bar} | {idx}/{total_count} | Elapsed: {format_time(elapsed)} | Remaining: {format_time(est_time)}", end="")

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

        # Show which file is currently being processed
        if input_path.is_dir():
            relative_path = image_path.relative_to(input_path)
        else:
            relative_path = image_path
        
        # Create progress bar
        progress_bar = create_progress_bar(idx, total_count)
        
        # Show detailed status
        status = f"\r{progress_bar} | {idx}/{total_count} | 🔄 {action} {relative_path}"
        if est_time != "Calculating...":
            status += f" | ⏱️  Elapsed: {format_time(elapsed)} | ⏳ Remaining: {est_time}"
        
        # Clear line and print status
        print(" " * 150, end="\r")
        print(status, end="")

        try:
            input_image = Image.open(image_path)

            # Convert RGBA images to RGB to avoid channel mismatch
            if input_image.mode == "RGBA":
                input_image = input_image.convert("RGB")

            # If in flat mode, copy the image to the output directory
            if args.flat:
                input_image.save(output_image_path)

            # Process each caption type
            # Collect all generated outputs before writing to file
            all_generated_outputs = []
            wrote_static_tags = False
            
            # Define task types that should be output as comma-separated tags
            tag_tasks = {
                "Booru tag list",
                "Booru-like tag list",
                "prompt_gen_tags", # Florence-2 tag generation
            }

            for task_idx, task in enumerate(args.caption_type, 1):
                # Show caption type progress
                if len(args.caption_type) > 1:
                    caption_progress = f" | Caption {task_idx}/{len(args.caption_type)}: {task}"
                    print(f"{caption_progress}", end="")
                
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
                elif task == "openai":
                    openai_prompt = args.openai_prompt
                    generated_output = caption_image_with_openai(
                        input_image,
                        args.openai_endpoint,
                        args.openai_api_key,
                        args.openai_model,
                        args.openai_prompt,
                        verbose=args.verbose,
                    )
                else:
                    print(f"Invalid caption type: {task}")
                    exit(1)

                # Post-process if it's a tag task
                if task in tag_tasks and generated_output:
                    # Ensure generated_output is a string
                    if isinstance(generated_output, list):
                        generated_output = generated_output[0] if generated_output else ""
                    
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

                # Collect the generated output if it's valid
                if generated_output:
                    # Ensure generated_output is a string
                    if isinstance(generated_output, list):
                        generated_output = generated_output[0] if generated_output else ""
                    
                    # Apply image name and path modifications
                    if args.include_image_name:
                        generated_output = f"{generated_output}, {image_path.name}"
                    if args.include_image_path:
                        path_segments = image_path.parts[-args.include_image_path:]
                        generated_output = f"{generated_output}, {', '.join(path_segments)}"
                    
                    all_generated_outputs.append(generated_output)

            # Write to file if we have valid generated content (do NOT write for static tags alone)
            if all_generated_outputs:
                with open(output_file_path, file_mode, encoding="utf-8") as output_file:
                    # Write static tags first if in write/overwrite mode and tags are provided
                    if file_mode == "w" and args.tags:
                        output_file.write(", ".join(args.tags))
                        wrote_static_tags = True

                    first_caption = True
                    for generated_output in all_generated_outputs:
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


        except Exception as e:
            error_count += 1
            print()  # Move to next line
            message = f"❌ Error: {image_path.name} - {str(e)}"
            print(message)

            # if there's only one image, raise the error
            if len(images) == 1:
                raise e
            else:
                continue

    # Calculate total elapsed time
    total_elapsed = time.time() - start_time

    # Print final summary
    print("\n" + "=" * 80)
    if input_path.is_dir():
        print(f"🎉 Finished processing images in {input_path}")
    else:
        print(f"🎉 Finished processing image {input_path}")
    
    print(f"📊 Summary:")
    print(f"   ✅ Processed: {processed_count}")
    print(f"   📝 Appended: {appended_count}")
    print(f"   ⏭️  Skipped: {skipped_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   📁 Total: {total_count}")
    print(f"   ⏱️  Total time: {format_time(total_elapsed)}")
    
    print(f"   📂 Output location: {args.output_path}")
    
    print("=" * 80)


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    
    # Handle create_config argument
    if args.create_config:
        create_sample_config(args.create_config)
        exit(0)

    # Load configuration from YAML file if specified
    if args.config:
        config = load_yaml_config(args.config)
        # Apply config values to args if they're not already set
        for key, value in config.items():
            if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

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

    # Display initial processing message
    print("🚀 AIPlay Image Captioner Starting...")
    print(f"📁 Input: {args.input_image}")
    print(f"🎯 Caption types: {', '.join(args.caption_type)}")
    if args.flat:
        print("📂 Output mode: Flat (sequential naming in ./output/)")
    print("⏳ Loading models and preparing to process files...")
    print("-" * 80)

    # Load models
    checkpoint_path = Path(args.checkpoint_path)

    clip_model = None
    tokenizer = None
    text_model = None
    image_adapter = None
    florence2_model = None
    florence2_processor = None

    # Validate OpenAI parameters if OpenAI caption type is selected
    if "openai" in args.caption_type:
        if not args.openai_endpoint:
            print("Error: --openai_endpoint is required when using 'openai' caption type")
            exit(1)
        if not args.openai_api_key:
            print("Error: --openai_api_key is required when using 'openai' caption type")
            exit(1)

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

    # Load OpenAI models (placeholder for consistency)
    if "openai" in args.caption_type:
        load_openai_models()

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
        
        # Create progress bar for duplicate checking
        def create_duplicate_progress_bar(current, total, width=40):
            progress = current / total if total > 0 else 0
            filled = int(width * progress)
            bar = "█" * filled + "░" * (width - filled)
            percentage = progress * 100
            return f"[{bar}] {percentage:.1f}%"
        
        for hash_idx, img_path in enumerate(image_files, 1):
            # Show progress for hash calculation
            progress_bar = create_duplicate_progress_bar(hash_idx, len(image_files))
            print(f"\r{progress_bar} | Checking hash {hash_idx}/{len(image_files)}: {img_path.name}", end="")
            
            file_hash = calculate_file_hash(img_path)
            if file_hash:
                if file_hash not in image_hashes:
                    image_hashes[file_hash] = []
                image_hashes[file_hash].append(img_path)
        
        print()  # Move to next line after progress bar

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
             print("✅ No duplicate images found.")
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
