# AIPlay Image Captioner

This script captions images using either the JoyTag/LLaMA model or the Florence-2 model, or both. It can process single images or entire directories recursively.

## Usage

```bash
python main.py [input_image] [options]
```

## Command-Line Options

### Positional Arguments

-   `input_image`
    -   **Type:** `str`
    -   **Required:** No (but script will exit if not provided and file/directory doesn't exist)
    -   **Description:** Path to the single image file or a directory containing images to be processed. If it's a directory, the script will search recursively for images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, case-insensitive).

### Optional Arguments

-   `--caption_type <type1> [<type2> ...]`
    -   **Type:** `str` (list)
    -   **Default:** `["Descriptive"]`
    -   **Description:** Specifies the type(s) of captions to generate. You can provide multiple types.
    -   **Choices:**
        -   **JoyTag/LLaMA Types:**
            -   `Descriptive`
            -   `Descriptive (Informal)`
            -   `Training Prompt`
            -   `MidJourney`
            -   `Booru tag list`
            -   `Booru-like tag list`
            -   `Art Critic`
            -   `Product Listing`
            -   `Social Media Post`
        -   **Florence-2 Types:**
            -   `region_caption`
            -   `dense_region_caption`
            -   `region_proposal`
            -   `caption`
            -   `detailed_caption`
            -   `more_detailed_caption`
            -   `caption_to_phrase_grounding`
            -   `referring_expression_segmentation`
            -   `ocr`
            -   `ocr_with_region`
            -   `docvqa`
            -   `prompt_gen_tags`
            -   `prompt_gen_mixed_caption`
            -   `prompt_gen_analyze`
            -   `prompt_gen_mixed_caption_plus`
        -   **OpenAI-Compatible Types:**
            -   `openai`

-   `--tags <tag1> [<tag2> ...]` / `-t <tag1> [<tag2> ...]`
    -   **Type:** `str` (list)
    -   **Default:** `[]`
    -   **Description:** Specifies static tags to be added at the very beginning of the output `.txt` file, separated by `, `. These tags are only added when creating a new file or overwriting an existing one (i.e., not when using `--append`).

-   `--caption_length <length>`
    -   **Type:** `str`
    -   **Default:** `"medium-length"`
    -   **Description:** Specifies the desired length for JoyTag/LLaMA captions. Can be a descriptive term or a specific word count (multiples of 10 from 20 to 260).
    -   **Choices:** `any`, `very short`, `short`, `medium-length`, `long`, `very long`, `20`, `30`, ..., `260`

-   `--extra_options <index1> [<index2> ...]`
    -   **Type:** `int` (list)
    -   **Default:** `[11, 14]` (Corresponds to "Do NOT mention any text..." and "Do NOT use any ambiguous language.")
    -   **Description:** Indices corresponding to extra instructions for customizing JoyTag/LLaMA captions. Use `--list_extra_options` to see the full list with indices.

-   `--name_input <name>`
    -   **Type:** `str`
    -   **Default:** `""`
    -   **Description:** If provided, instructs the JoyTag/LLaMA model to refer to any person/character in the image using this name (requires extra option index 0 to be included in `--extra_options`).

-   `--custom_prompt <prompt>`
    -   **Type:** `str`
    -   **Default:** `""`
    -   **Description:** A completely custom prompt to use for caption generation. If provided, this overrides `--caption_type`, `--caption_length`, `--extra_options`, and `--name_input`. Note: Ensure your custom prompt is compatible with the selected model(s).

-   `--checkpoint_path <path>`
    -   **Type:** `str`
    -   **Default:** `"models"`
    -   **Description:** Path to the directory where model checkpoints are stored or will be downloaded.

-   `--llm_model_id <model_id>`
    -   **Type:** `str`
    -   **Default:** `"unsloth/Meta-Llama-3.1-8B-Instruct"`
    -   **Description:** The Hugging Face model ID for the JoyTag/LLaMA text model component.

-   `--florence2_model_id <model_id>`
    -   **Type:** `str`
    -   **Default:** `"MiaoshouAI/Florence-2-large-PromptGen-v2.0"`
    -   **Description:** The Hugging Face model ID for the Florence-2 model.

-   `--list_extra_options`
    -   **Action:** `store_true`
    -   **Description:** Prints the list of available extra options for JoyTag/LLaMA captions along with their corresponding indices, then exits.

-   `-o`, `--overwrite`
    -   **Action:** `store_true`
    -   **Description:** If a caption file (`.txt`) already exists for an image, overwrite it. By default, existing files are skipped.

-   `-a`, `--append`
    -   **Action:** `store_true`
    -   **Description:** If a caption file (`.txt`) already exists, append the new caption(s) to it (separated by two newlines). Cannot be used with `--overwrite`. If both are specified, the default skip behavior applies.

-   `-v`, `--verbose`
    -   **Action:** `store_true`
    -   **Description:** Show more detailed information during processing, including model loading details and intermediate steps if applicable.

-   `--openai_endpoint <url>`
    -   **Type:** `str`
    -   **Default:** `""`
    -   **Description:** URL of the OpenAI-compatible API endpoint (required when using `openai` caption type).

-   `--openai_api_key <key>`
    -   **Type:** `str`
    -   **Default:** `""`
    -   **Description:** API key for the OpenAI-compatible endpoint (required when using `openai` caption type).

-   `--openai_model <model>`
    -   **Type:** `str`
    -   **Default:** `"gpt-4o"`
    -   **Description:** Model name to use with the OpenAI-compatible API.

-   `--openai_prompt <prompt>`
    -   **Type:** `str`
    -   **Default:** `"Please describe this image in detail."`
    -   **Description:** Custom prompt to send with the image to the OpenAI-compatible API.

## Examples

**Caption a single image with default descriptive settings:**

```bash
python main.py path/to/your/image.jpg
```

**Generate Booru tags and a detailed Florence-2 caption for all images in a directory, overwriting existing captions:**

```bash
python main.py path/to/image/directory --caption_type "Booru tag list" detailed_caption -o
```

**Generate a short, informal descriptive caption for an image, referring to the character as "Alice", and providing specific extra instructions:**

```bash
python main.py character.png --caption_type "Descriptive (Informal)" --caption_length short --name_input "Alice" --extra_options 0 2 12
```

**Generate Florence-2 OCR results for an image:**

```bash
python main.py document.png --caption_type ocr
```

**Generate captions using an OpenAI-compatible API:**

```bash
python main.py image.jpg --caption_type openai --openai_endpoint "https://api.openai.com/v1/chat/completions" --openai_api_key "your-api-key-here"
```

**Generate captions using a custom OpenAI-compatible endpoint with a specific model and prompt:**

```bash
python main.py images/ --caption_type openai --openai_endpoint "https://your-endpoint.com/v1/chat/completions" --openai_api_key "your-key" --openai_model "gpt-4-vision-preview" --openai_prompt "Describe this image in a creative and engaging way."
```

**List available extra options:**

```bash
python main.py --list_extra_options
```