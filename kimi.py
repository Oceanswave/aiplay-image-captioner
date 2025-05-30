from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from modules.utils import download_hg_model

kimi_model_id = "moonshotai/Kimi-VL-A3B-Instruct"

kimi_model_checkpoint = download_hg_model("models", kimi_model_id)

model = AutoModelForCausalLM.from_pretrained(
    kimi_model_checkpoint,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(kimi_model_checkpoint, trust_remote_code=True)

image_path = "./image.png"
image = Image.open(image_path)
messages = [
    {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": "What is the dome building in the picture? Think step by step."}]}
]
text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
response = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(response)
