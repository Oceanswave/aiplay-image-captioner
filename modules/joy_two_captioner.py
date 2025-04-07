from pathlib import Path
from PIL import Image
from typing import Union, List

from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
)
import torch
import torchvision.transforms.functional as TVF

from modules.joy_caption_two import CAPTION_TYPE_MAP
from modules.utils import download_hg_model, ImageAdapter


def load_models(
    device: torch.device,
    torch_dtype: torch.dtype,
    models_dir: str,
    llm_model_id: str,
    verbose: bool = False,
):
    base_model_path = Path(models_dir, "Joy_caption_two")

    # Load CLIP
    model_id = "google/siglip-so400m-patch14-384"
    clip_path = download_hg_model(models_dir, model_id, "clip")

    if verbose:
        print("Loading CLIP model...")

    clip_model = AutoModel.from_pretrained(
        clip_path, trust_remote_code=True, torch_dtype=torch_dtype
    )

    clip_model = clip_model.vision_model

    assert (base_model_path / "clip_model.pt").exists()
    if verbose:
        print("Loading VLM's custom vision model")
    checkpoint = torch.load(
        base_model_path / "clip_model.pt", map_location="cpu", weights_only=True
    )
    checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
    clip_model.load_state_dict(checkpoint)
    del checkpoint

    clip_model.eval()
    clip_model.requires_grad_(False)
    clip_model.to(device)

    # Tokenizer
    if verbose:
        print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path / "text_model", use_fast=True
    )
    assert isinstance(
        tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)
    ), f"Tokenizer is of type {type(tokenizer)}"

    # LLM
    if verbose:
        print("Loading LLM")
        print("Loading VLM's custom text model")
    llm_model_checkpoint = download_hg_model(models_dir, llm_model_id, "LLM")
    assert (base_model_path / "clip_model.pt").exists()
    if verbose:
        print(f"Loading VLM's custom vision model from {llm_model_checkpoint}")
    text_model = AutoModelForCausalLM.from_pretrained(
        llm_model_checkpoint, torch_dtype=torch_dtype
    )
    text_model.eval()
    text_model.to(device)

    # Image Adapter
    if verbose:
        print("Loading image adapter")
    image_adapter = ImageAdapter(
        clip_model.config.hidden_size,
        text_model.config.hidden_size,
        False,
        False,
        38,
        False,
    )
    image_adapter.load_state_dict(
        torch.load(base_model_path / "image_adapter.pt", map_location="cpu")
    )
    image_adapter.eval()
    image_adapter.to(device)

    return clip_model, tokenizer, text_model, image_adapter


@torch.no_grad()
def caption_image(
    device: torch.device,
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
    verbose: bool = False,
    autocast: torch.amp.autocast = None,
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
    prompt_str = prompt_str.format(
        name=name_input, length=caption_length, word_count=caption_length
    )

    if custom_prompt.strip() != "":
        prompt_str = custom_prompt.strip()

    # For debugging
    if verbose:
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
        vision_outputs = clip_model(
            pixel_values=pixel_values, output_hidden_states=True
        )
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
            "<|system|>"
            + convo[0]["content"]
            + "<|end|><|user|>"
            + convo[1]["content"]
            + "<|end|>"
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
    assert isinstance(convo_tokens, torch.Tensor) and isinstance(
        prompt_tokens, torch.Tensor
    )
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
    preamble_len = (
        eot_id_indices[1] - prompt_tokens.shape[0]
    )  # Number of tokens before the prompt

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
            torch.zeros(
                (1, embedded_images.shape[1]), dtype=torch.long
            ),  # Dummy tokens for the image
            convo_tokens[preamble_len:].unsqueeze(0),
        ],
        dim=1,
    ).to(device)
    attention_mask = torch.ones_like(input_ids)

    # Debugging
    if verbose:
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
    generate_ids = generate_ids[:, input_ids.shape[1] :]
    if generate_ids[0][-1] in [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]:
        generate_ids = generate_ids[:, :-1]

    caption = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]

    return prompt_str, caption.strip()
