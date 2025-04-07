Image Captioner
---
Standalone VLM image captioner using [JoyCaption](https://github.com/fpgaminer/joycaption) and [Florence 2] based off the [SLK ComfyUI Implementation](https://github.com/EvilBT/ComfyUI_SLK_joy_caption_two) and [Kiljai Florence2](https://github.com/kijai/ComfyUI-Florence2) implementations


Example Usage:

Process a folder of images and generates caption files with both Booru and SDXL-based captions, overwriting existing caption files
```
python main.py ./my-images/ --caption_type 'Training Prompt' -o --extra_options 0 6 11 14
```

process a single file and append caption files with MiaoshouAI's Prompt Gen tags
```
python main.py ./image.png --caption_type "prompt_gen_tags" -a
```