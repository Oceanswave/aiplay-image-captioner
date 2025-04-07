from typing import List
import torch
import torchvision.transforms.functional as F
from transformers import AutoProcessor, AutoModelForCausalLM
import os
import io
import re
import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageColor, ImageFont

from modules.florence_two import TASK_TYPE_MAP, COLOR_MAP
from modules.utils import download_hg_model

# workaround for unnecessary flash_attn requirement
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    try:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
    except:
        print(f"No flash_attn import to remove")
        pass
    return imports


def load_models(
    device: torch.device,
    dtype: torch.dtype,
    models_dir: str,
    florence_model_id: str,
    _: bool = False,
):
    florence_model_checkpoint = download_hg_model(models_dir, florence_model_id)
    attention = "sdpa"
    print(f"Florence2 using {attention} for attention")
    with patch(
        "transformers.dynamic_module_utils.get_imports", fixed_get_imports
    ):  # workaround for unnecessary flash_attn requirement
        model = AutoModelForCausalLM.from_pretrained(
            florence_model_checkpoint,
            attn_implementation=attention,
            device_map=device,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    processor = AutoProcessor.from_pretrained(
        florence_model_checkpoint, trust_remote_code=True
    )

    return model, processor


def caption_images(
    device: torch.device,
    torch_dtype: torch.dtype,
    images: List[Image.Image],
    text_input: str,
    task: str,
    model,
    processor,
    max_new_tokens: int = 4096,
    num_beams: int = 3,
    do_sample: bool = False,
    fill_mask: bool = True,
    output_mask_select: str = "",
    verbose: bool = False,
):
    task_prompt = TASK_TYPE_MAP.get(task, "<OD>")

    if (
        task
        not in [
            "referring_expression_segmentation",
            "caption_to_phrase_grounding",
            "docvqa",
        ]
    ) and text_input:
        raise ValueError(
            "Text input (prompt) is only supported for 'referring_expression_segmentation', 'caption_to_phrase_grounding', and 'docvqa'"
        )

    if text_input != "":
        prompt = task_prompt + " " + text_input
    else:
        prompt = task_prompt

    out = []
    out_masks = []
    out_results = []
    out_data = []

    for img in images:
        image_pil = img
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        inputs = processor(
            text=prompt, images=image_pil, return_tensors="pt", do_rescale=False
        ).to(device, torch_dtype)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
        )

        results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        if verbose:
            print(results)

        # cleanup the special tokens from the final list
        if task == "ocr_with_region":
            clean_results = str(results)
            cleaned_string = re.sub(r"</?s>|<[^>]*>", "\n", clean_results)
            clean_results = re.sub(r"\n+", "\n", cleaned_string)
        else:
            clean_results = str(results)
            clean_results = clean_results.replace("</s>", "")
            clean_results = clean_results.replace("<s>", "")

        if len(images) == 1:
            out_results = clean_results
        else:
            out_results.append(clean_results)

        W, H = image_pil.size

        parsed_answer = processor.post_process_generation(
            results, task=task_prompt, image_size=(W, H)
        )

        if (
            task == "region_caption"
            or task == "dense_region_caption"
            or task == "caption_to_phrase_grounding"
            or task == "region_proposal"
        ):
            fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            ax.imshow(image_pil)
            bboxes = parsed_answer[task_prompt]["bboxes"]
            labels = parsed_answer[task_prompt]["labels"]

            mask_indexes = []
            # Determine mask indexes outside the loop
            if output_mask_select != "":
                mask_indexes = [n for n in output_mask_select.split(",")]
                print(mask_indexes)
            else:
                mask_indexes = [str(i) for i in range(len(bboxes))]

            # Initialize mask_layer only if needed
            if fill_mask:
                mask_layer = Image.new("RGB", image_pil.size, (0, 0, 0))
                mask_draw = ImageDraw.Draw(mask_layer)

            for index, (bbox, label) in enumerate(zip(bboxes, labels)):
                # Modify the label to include the index
                indexed_label = f"{index}.{label}"

                if fill_mask:
                    if str(index) in mask_indexes:
                        print(
                            "match index:", str(index), "in mask_indexes:", mask_indexes
                        )
                        mask_draw.rectangle(
                            [bbox[0], bbox[1], bbox[2], bbox[3]], fill=(255, 255, 255)
                        )
                    if label in mask_indexes:
                        print("match label")
                        mask_draw.rectangle(
                            [bbox[0], bbox[1], bbox[2], bbox[3]], fill=(255, 255, 255)
                        )

                # Create a Rectangle patch
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]),  # (x,y) - lower left corner
                    bbox[2] - bbox[0],  # Width
                    bbox[3] - bbox[1],  # Height
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                    label=indexed_label,
                )
                # Calculate text width with a rough estimation
                text_width = len(label) * 6  # Adjust multiplier based on your font size
                text_height = 12  # Adjust based on your font size

                # Initial text position
                text_x = bbox[0]
                text_y = (
                    bbox[1] - text_height
                )  # Position text above the top-left of the bbox

                # Adjust text_x if text is going off the left or right edge
                if text_x < 0:
                    text_x = 0
                elif text_x + text_width > W:
                    text_x = W - text_width

                # Adjust text_y if text is going off the top edge
                if text_y < 0:
                    text_y = bbox[
                        3
                    ]  # Move text below the bottom-left of the bbox if it doesn't overlap with bbox

                # Add the rectangle to the plot
                ax.add_patch(rect)
                facecolor = random.choice(COLOR_MAP) if len(images) == 1 else "red"
                # Add the label
                plt.text(
                    text_x,
                    text_y,
                    indexed_label,
                    color="white",
                    fontsize=12,
                    bbox=dict(facecolor=facecolor, alpha=0.5),
                )
            if fill_mask:
                mask_tensor = F.to_tensor(mask_layer)
                mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
                mask_tensor = mask_tensor.repeat(1, 1, 1, 3)
                mask_tensor = mask_tensor[:, :, :, 0]
                out_masks.append(mask_tensor)

            # Remove axis and padding around the image
            ax.axis("off")
            ax.margins(0, 0)
            ax.get_xaxis().set_major_locator(plt.NullLocator())
            ax.get_yaxis().set_major_locator(plt.NullLocator())
            fig.canvas.draw()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", pad_inches=0)
            buf.seek(0)
            annotated_image_pil = Image.open(buf)

            annotated_image_tensor = F.to_tensor(annotated_image_pil)
            out_tensor = (
                annotated_image_tensor[:3, :, :]
                .unsqueeze(0)
                .permute(0, 2, 3, 1)
                .cpu()
                .float()
            )
            out.append(out_tensor)

            if task == "caption_to_phrase_grounding":
                out_data.append(parsed_answer[task_prompt])
            else:
                out_data.append(bboxes)

            plt.close(fig)

        elif task == "referring_expression_segmentation":
            # Create a new black image
            mask_image = Image.new("RGB", (W, H), "black")
            mask_draw = ImageDraw.Draw(mask_image)

            predictions = parsed_answer[task_prompt]

            # Iterate over polygons and labels
            for polygons, label in zip(predictions["polygons"], predictions["labels"]):
                color = random.choice(COLOR_MAP)
                for _polygon in polygons:
                    _polygon = np.array(_polygon).reshape(-1, 2)
                    # Clamp polygon points to image boundaries
                    _polygon = np.clip(_polygon, [0, 0], [W - 1, H - 1])
                    if len(_polygon) < 3:
                        print("Invalid polygon:", _polygon)
                        continue

                    _polygon = _polygon.reshape(-1).tolist()

                    # Draw the polygon
                    if fill_mask:
                        overlay = Image.new("RGBA", image_pil.size, (255, 255, 255, 0))
                        image_pil = image_pil.convert("RGBA")
                        draw = ImageDraw.Draw(overlay)
                        color_with_opacity = ImageColor.getrgb(color) + (180,)
                        draw.polygon(
                            _polygon, outline=color, fill=color_with_opacity, width=3
                        )
                        image_pil = Image.alpha_composite(image_pil, overlay)
                    else:
                        draw = ImageDraw.Draw(image_pil)
                        draw.polygon(_polygon, outline=color, width=3)

                    # draw mask
                    mask_draw.polygon(_polygon, outline="white", fill="white")

            image_tensor = F.to_tensor(image_pil)
            image_tensor = (
                image_tensor[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
            )
            out.append(image_tensor)

            mask_tensor = F.to_tensor(mask_image)
            mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
            mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
            mask_tensor = mask_tensor.repeat(1, 1, 1, 3)
            mask_tensor = mask_tensor[:, :, :, 0]
            out_masks.append(mask_tensor)

        elif task == "ocr_with_region":
            try:
                font = ImageFont.load_default().font_variant(size=24)
            except:
                font = ImageFont.load_default()
            predictions = parsed_answer[task_prompt]
            scale = 1
            image_pil = image_pil.convert("RGBA")
            overlay = Image.new("RGBA", image_pil.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            bboxes, labels = predictions["quad_boxes"], predictions["labels"]

            # Create a new black image for the mask
            mask_image = Image.new("RGB", (W, H), "black")
            mask_draw = ImageDraw.Draw(mask_image)

            for box, label in zip(bboxes, labels):
                scaled_box = [
                    v / (W if idx % 2 == 0 else H) for idx, v in enumerate(box)
                ]
                out_data.append({"label": label, "box": scaled_box})

                color = random.choice(COLOR_MAP)
                new_box = (np.array(box) * scale).tolist()

                if fill_mask:
                    color_with_opacity = ImageColor.getrgb(color) + (180,)
                    draw.polygon(
                        new_box, outline=color, fill=color_with_opacity, width=3
                    )
                else:
                    draw.polygon(new_box, outline=color, width=3)

                draw.text(
                    (new_box[0] + 8, new_box[1] + 2),
                    "{}".format(label),
                    align="right",
                    font=font,
                    fill=color,
                )

                # Draw the mask
                mask_draw.polygon(new_box, outline="white", fill="white")

            image_pil = Image.alpha_composite(image_pil, overlay)
            image_pil = image_pil.convert("RGB")

            image_tensor = F.to_tensor(image_pil)
            image_tensor = (
                image_tensor[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
            )
            out.append(image_tensor)

            # Process the mask
            mask_tensor = F.to_tensor(mask_image)
            mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
            mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
            mask_tensor = mask_tensor.repeat(1, 1, 1, 3)
            mask_tensor = mask_tensor[:, :, :, 0]
            out_masks.append(mask_tensor)

        elif task == "docvqa":
            if text_input == "":
                raise ValueError("Text input (prompt) is required for 'docvqa'")
            prompt = "<DocVQA> " + text_input

            inputs = (
                processor(
                    text=prompt, images=image_pil, return_tensors="pt", do_rescale=False
                )
                .to(torch_dtype)
                .to(device)
            )
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
            )

            results = processor.batch_decode(generated_ids, skip_special_tokens=False)[
                0
            ]
            clean_results = results.replace("</s>", "").replace("<s>", "")

            if len(images) == 1:
                out_results = clean_results
            else:
                out_results.append(clean_results)

            out.append(
                F.to_tensor(image_pil).unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
            )

    if len(out) > 0:
        out_tensor = torch.cat(out, dim=0)
    else:
        out_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
    if len(out_masks) > 0:
        out_mask_tensor = torch.cat(out_masks, dim=0)
    else:
        out_mask_tensor = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")

    return (out_tensor, out_mask_tensor, out_results, out_data)
