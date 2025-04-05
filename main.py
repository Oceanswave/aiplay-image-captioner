import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
)
from PIL import Image
import os
import torchvision.transforms.functional as TVF
import contextlib
from typing import Union, List

from utils import download_hg_model

CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a descriptive caption for this image in a formal tone.",
        "Write a descriptive caption for this image in a formal tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a formal tone.",
    ],
    "Descriptive (Informal)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Training Prompt": [
        "Write a stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image within {word_count} words.",
        "Write a {length} stable diffusion prompt for this image.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Booru tag list": [
        "Write a list of Booru tags for this image.",
        "Write a list of Booru tags for this image within {word_count} words.",
        "Write a {length} list of Booru tags for this image.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

EXTRA_OPTIONS_LIST = [
    "If there is a person/character in the image you must refer to them as {name}.",
    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
    "Include information about lighting.",
    "Include information about camera angle.",
    "Include information about whether there is a watermark or not.",
    "Include information about whether there are JPEG artifacts or not.",
    "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
    "Do NOT include anything sexual; keep it PG.",
    "Do NOT mention the image's resolution.",
    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
    "Do NOT mention any text that is in the image.",
    "Specify the depth of field and whether the background is in focus or blurred.",
    "If applicable, mention the likely use of artificial or natural lighting sources.",
    "Do NOT use any ambiguous language.",
    "Include whether the image is sfw, suggestive, or nsfw.",
    "ONLY describe the most important elements of the image.",
]

CAPTION_LENGTH_CHOICES = (
    ["any", "very short", "short", "medium-length", "long", "very long"]
    + [str(i) for i in range(20, 261, 10)]
)

HF_TOKEN = os.environ.get("HF_TOKEN", None)

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

class ImageAdapter(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        ln1: bool,
        pos_emb: bool,
        num_image_tokens: int,
        deep_extract: bool,
    ):
        super().__init__()
        self.deep_extract = deep_extract

        if self.deep_extract:
            input_features = input_features * 5

        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = (
            None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))
        )

        # Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(
            mean=0.0, std=0.02
        )  # Matches HF's implementation of llama3

    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.concat(
                (
                    vision_outputs[-2],
                    vision_outputs[3],
                    vision_outputs[7],
                    vision_outputs[13],
                    vision_outputs[20],
                ),
                dim=-1,
            )
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"  # batch, tokens, features
            assert (
                x.shape[-1] == vision_outputs[-2].shape[-1] * 5
            ), f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)

        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        # <|image_start|>, IMAGE, <|image_end|>
        other_tokens = self.other_tokens(
            torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1)
        )
        assert other_tokens.shape == (
            x.shape[0],
            2,
            x.shape[2],
        ), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)


# Determine the device to use (GPU, MPS, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch_dtype = torch.bfloat16
    autocast = lambda: torch.amp.autocast(device_type='cuda')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    torch_dtype = torch.float32  # MPS doesn't support bfloat16
    autocast = contextlib.nullcontext  # No autocasting on MPS
else:
    device = torch.device("cpu") 
    torch_dtype = torch.float32
    autocast = contextlib.nullcontext  # No autocasting on CPU

print(f"Using device: {device}")


def load_models(models_dir, llm_model_id):
    base_model_path = Path(models_dir, "Joy_caption_two")

    # Load CLIP
    model_id = "google/siglip-so400m-patch14-384"
    clip_path = download_hg_model(models_dir, model_id, "clip")

    if args.verbose:
        print("Loading CLIP model...")
    
    clip_model = AutoModel.from_pretrained(
        clip_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype
    )

    clip_model = clip_model.vision_model

    assert (base_model_path / "clip_model.pt").exists()
    if args.verbose:
        print("Loading VLM's custom vision model")
    checkpoint = torch.load(base_model_path / "clip_model.pt", map_location='cpu', weights_only=True)
    checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
    clip_model.load_state_dict(checkpoint)
    del checkpoint

    clip_model.eval()
    clip_model.requires_grad_(False)
    clip_model.to(device)

    # Tokenizer
    if args.verbose:
        print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path / "text_model", use_fast=True
    )
    assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)), f"Tokenizer is of type {type(tokenizer)}"

    # LLM
    if args.verbose:
        print("Loading LLM")
        print("Loading VLM's custom text model")
    llm_model_checkpoint = download_hg_model(models_dir, llm_model_id, "LLM")
    assert (base_model_path / "clip_model.pt").exists()
    if args.verbose:
        print(f"Loading VLM's custom vision model from {llm_model_checkpoint}")
    text_model = AutoModelForCausalLM.from_pretrained(
        llm_model_checkpoint, torch_dtype=torch_dtype
    )
    text_model.eval()
    text_model.to(device)

    # Image Adapter
    if args.verbose:
        print("Loading image adapter")
    image_adapter = ImageAdapter(
        clip_model.config.hidden_size, text_model.config.hidden_size, False, False, 38, False
    )
    image_adapter.load_state_dict(
        torch.load(base_model_path / "image_adapter.pt", map_location="cpu")
    )
    image_adapter.eval()
    image_adapter.to(device)

    return clip_model, tokenizer, text_model, image_adapter


@torch.no_grad()
def stream_chat(
    input_image: Image.Image,
    caption_type: str,
    caption_length: Union[str, int],
    extra_options: List[str],
    name_input: str,
    custom_prompt: str,
    clip_model,
    tokenizer,
    text_model,
    image_adapter,
) -> tuple:
    if device.type == "cuda" or device.type == "mps":
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    # 'any' means no length specified
    length = None if caption_length == "any" else caption_length

    if isinstance(length, str):
        try:
            length = int(length)
        except ValueError:
            pass

    # Build prompt
    if length is None:
        map_idx = 0
    elif isinstance(length, int):
        map_idx = 1
    elif isinstance(length, str):
        map_idx = 2
    else:
        raise ValueError(f"Invalid caption length: {length}")

    prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]

    # Add extra options
    if len(extra_options) > 0:
        prompt_str += " " + " ".join(extra_options)

    # Add name, length, word_count
    prompt_str = prompt_str.format(name=name_input, length=caption_length, word_count=caption_length)

    if custom_prompt.strip() != "":
        prompt_str = custom_prompt.strip()

    # For debugging
    if args.verbose:
        print(f"Prompt: {prompt_str}")

    # Preprocess image
    # NOTE: I found the default processor for so400M to have worse results than just using PIL directly
    # image = clip_processor(images=input_image, return_tensors='pt').pixel_values
    if input_image.mode == "RGBA":
        input_image = input_image.convert("RGB")
    
    image = input_image.resize((384, 384), Image.LANCZOS)
    pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
    pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
    pixel_values = pixel_values.to(device)

    # Embed image
    # This results in Batch x Image Tokens x Features
    with autocast():
        vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
        embedded_images = image_adapter(vision_outputs.hidden_states)
        embedded_images = embedded_images.to(device)

    # Build the conversation
    convo = [
        {
            "role": "system",
            "content": "You are a helpful image captioner.",
        },
        {
            "role": "user",
            "content": prompt_str,
        },
    ]

    # Format the conversation
    # The apply_chat_template method might not be available; handle accordingly
    if hasattr(tokenizer, "apply_chat_template"):
        convo_string = tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )
    else:
        # Simple concatenation if apply_chat_template is not available
        convo_string = (
            "<|system|>" + convo[0]["content"] + "<|end|><|user|>" + convo[1]["content"] + "<|end|>"
        )

    assert isinstance(convo_string, str)

    # Tokenize the conversation
    # prompt_str is tokenized separately so we can do the calculations below
    convo_tokens = tokenizer.encode(
        convo_string, return_tensors="pt", add_special_tokens=False, truncation=False
    )
    prompt_tokens = tokenizer.encode(
        prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False
    )
    assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
    convo_tokens = convo_tokens.squeeze(0)  # Squeeze just to make the following easier
    prompt_tokens = prompt_tokens.squeeze(0)

    # Calculate where to inject the image
    eot_id_indices = (
        (convo_tokens == tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        .nonzero(as_tuple=True)[0]
        .tolist()
    )
    if len(eot_id_indices) != 2:
        # Fallback if <|eot_id|> tokens are not present
        eot_id_indices = [0, len(convo_tokens)]
    preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]  # Number of tokens before the prompt

    # Embed the tokens
    convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to(device))

    # Construct the input
    input_embeds = torch.cat(
        [
            convo_embeds[:, :preamble_len],  # Part before the prompt
            embedded_images.to(dtype=convo_embeds.dtype),  # Image
            convo_embeds[:, preamble_len:],  # The prompt and anything after it
        ],
        dim=1,
    ).to(device)

    input_ids = torch.cat(
        [
            convo_tokens[:preamble_len].unsqueeze(0),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),  # Dummy tokens for the image
            convo_tokens[preamble_len:].unsqueeze(0),
        ],
        dim=1,
    ).to(device)
    attention_mask = torch.ones_like(input_ids)

    # Debugging
    if args.verbose:
        print(f"Input to model: {repr(tokenizer.decode(input_ids[0]))}")

    # Generate the caption
    generate_ids = text_model.generate(
        input_ids,
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        suppress_tokens=None,
    )

    # Trim off the prompt
    generate_ids = generate_ids[:, input_ids.shape[1]:]
    if generate_ids[0][-1] in [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]:
        generate_ids = generate_ids[:, :-1]

    caption = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]

    return prompt_str, caption.strip()


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


    input_image_type = "single"
    if args.input_image and Path(args.input_image).is_dir():
        input_image_type = "directory"

    name_input = args.name_input
    custom_prompt = args.custom_prompt
    checkpoint_path = Path(args.checkpoint_path)

    # test to see if the input_image exists and is a file or directory
    if not args.input_image or not Path(args.input_image).exists():
        print(f"Input image not found: {args.input_image}")
        exit(1)

    # Load models
    (
        clip_model,
        tokenizer,
        text_model,
        image_adapter,
    ) = load_models(checkpoint_path, args.llm_model_id)

    input_image_type = "single"
    if args.input_image and Path(args.input_image).is_dir():
        input_image_type = "directory"

    # Process single image
    if input_image_type == "single":
        input_image_path = Path(args.input_image)
        if not input_image_path.is_file():
            print(f"Input image not found: {input_image_path}")
            exit(1)
        
        # Create output file path with same name but .txt extension
        output_file_path = input_image_path.with_suffix('.txt')
        
        # Check if output file exists
        file_exists = output_file_path.exists()
        
        # Determine file mode based on options
        action = ""
        file_mode = "w"  # Default is write (create or overwrite)
        if file_exists:
            if args.append:
                file_mode = "a"  # Append to existing file
                action = "Appending"
                appended_count = 1
            elif args.overwrite:
                action = "Overwriting"
                processed_count = 1
            else:
                if not args.verbose:
                    # Print on a new line to keep history
                    print(f"Skipped: {input_image_path.name} (already exists)")
                skipped_count = 1
                # Show current progress on the same line
                progress = f"Progress: 1 processed, 0 appended, 1 skipped, 0 errors"
                padding = " " * 20  # Add extra spaces to ensure previous output is cleared
                print(f"{progress}{padding}", end="\r")
                exit(0)
        else:
            action = "Processing"
            processed_count = 1
        
        # Show which file is currently being processed (with carriage return)
        if not args.verbose:
            # Show file being worked on with carriage return (overwrites the line)
            relative_path = input_image_path.relative_to(input_image_path.parent)
            status = f"Working: {action} {relative_path} (1/1)"
            padding = " " * 30
            print(f"{status}{padding}", end="\r")
        
        try:
            input_image = Image.open(input_image_path)
            
            # Convert RGBA images to RGB to avoid channel mismatch
            if input_image.mode == "RGBA":
                input_image = input_image.convert("RGB")
            
            # Process each caption type
            with open(output_file_path, file_mode, encoding="utf-8") as output_file:
                for caption_type in args.caption_type:
                    if args.verbose:
                        print(f"Generating {caption_type} caption...")
                    
                    prompt_str, caption = stream_chat(
                        input_image,
                        caption_type,
                        args.caption_length,
                        extra_options,
                        name_input,
                        custom_prompt,
                        clip_model,
                        tokenizer,
                        text_model,
                        image_adapter,
                    )
                    
                    # Write caption to individual text file
                    output_file.write(f"{caption}\n\n")
            
            # Print completed file info on a new line to keep history
            if not args.verbose:
                print(f"Complete: {action} {input_image_path.name}")
            else:
                print(f"Captions saved to {output_file_path}")
            
        except Exception as e:
            error_count = 1
            print(f"\nError: {input_image_path.name} - {str(e)}")
            exit(1)
        
        # Show current progress (overwrites the line)
        if not args.verbose:
            progress = f"Progress: 1 processed, 0 appended, 1 skipped, 0 errors"
            padding = " " * 30
            print(f"{progress}{padding}", end="\r")

    # Process images in a directory
    elif input_image_type == "directory":
        input_dir = Path(args.input_image)
        if not input_dir.is_dir():
            print(f"Input directory not found: {input_dir}")
            exit(1)

        # List of image file extensions
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]

        # Collect all image files in the directory and subdirectories recursively
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"**/*{ext}"))
            image_files.extend(input_dir.glob(f"**/*{ext.upper()}"))  # Also include uppercase extensions

        if not image_files:
            print(f"No image files found in directory: {input_dir} (including subdirectories)")
            exit(1)
        
        print(f"Found {len(image_files)} images in {input_dir} and its subdirectories")

        processed_count = 0
        skipped_count = 0
        appended_count = 0
        error_count = 0
        total_count = len(image_files)
        
        for idx, image_path in enumerate(image_files, 1):
            # Create output file path with same name but .txt extension
            output_file_path = image_path.with_suffix('.txt')
            
            # Check if output file exists
            file_exists = output_file_path.exists()
            
            # Determine file mode based on options
            action = ""
            file_mode = "w"  # Default is write (create or overwrite)
            if file_exists:
                if args.append:
                    file_mode = "a"  # Append to existing file
                    action = "Appending to"
                    appended_count += 1
                elif args.overwrite:
                    action = "Overwriting"
                    processed_count += 1
                else:
                    # Print skipped files on a new line to maintain history
                    print(f"Skipped: {image_path.name} (file already exists)")
                    skipped_count += 1
                    continue
            else:
                action = "Processing"
                processed_count += 1
            
            # Show which file is currently being processed (with carriage return)
            relative_path = image_path.relative_to(input_dir)
            status = f"Working: {action} {relative_path} ({idx}/{total_count})"
            padding = " " * 30  # Add extra spaces to ensure previous output is cleared
            print(f"{status}{padding}", end="\r")
            
            try:
                input_image = Image.open(image_path)
                
                # Convert RGBA images to RGB to avoid channel mismatch
                if input_image.mode == "RGBA":
                    input_image = input_image.convert("RGB")
                
                # Process each caption type
                with open(output_file_path, file_mode, encoding="utf-8") as output_file:
                    for caption_type in args.caption_type:
                        if args.verbose:
                            print(f"Generating {caption_type} caption...")
                        
                        prompt_str, caption = stream_chat(
                            input_image,
                            caption_type,
                            args.caption_length,
                            extra_options,
                            name_input,
                            custom_prompt,
                            clip_model,
                            tokenizer,
                            text_model,
                            image_adapter,
                        )
                        
                        # Write caption to individual text file
                        output_file.write(f"{caption}\n\n")
                
                # Print completed file on a new line to maintain history
                print(f"Completed: {action} {image_path.name}")
                
            except Exception as e:
                error_count += 1
                print(f"Error: {image_path.name} - {str(e)}")
                continue
        
        # Print summary at the end
        print(f"\nFinished processing images in {input_dir}")
        print(f"Processed: {processed_count}, Appended: {appended_count}, Skipped: {skipped_count}, Errors: {error_count}, Total: {total_count}")
    else:
        print(f"Invalid input image type: {input_image_type}")
        exit(1)
