Image Captioner
---
Standalone VLM image captioner using [JoyCaption](https://github.com/fpgaminer/joycaption) based off the [SLK ComfyUI Implementation](https://github.com/EvilBT/ComfyUI_SLK_joy_caption_two)


Example Usage:

Process a folder of images and generates caption files with both Booru and SDXL-based captions, overwriting existing caption files
```
python main.py ./my-images/ --caption_type 'Training Prompt' -o --extra_options 0 6 11 14
```
