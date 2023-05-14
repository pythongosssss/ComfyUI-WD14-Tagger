# ComfyUI WD 1.4 Tagger

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom extension allowing the interrogation of booru tags from images.

Based on [SmilingWolf/wd-v1-4-tags](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags) and [toriato/stable-diffusion-webui-wd14-tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger)  
All models created by [SmilingWolf](https://huggingface.co/SmilingWolf)

## Installation
1. `git clone https://github.com/pythongosssss/ComfyUI-WD14-Tagger` into the `custom_nodes` folder 
    - e.g. `custom_nodes\ComfyUI-WD14-Tagger`  
2. Open a Command Prompt/Terminal/PowerShell/etc
3. Change to the `custom_nodes\ComfyUI-WD14-Tagger` folder you just created 
    - e.g. `cd C:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-WD14-Tagger` or wherever you have it installed
4.  Install python packages
      - **Standalone installation** (embedded python):   
       `../../../python_embeded/python.exe -s -m pip install -r requirements.txt`  
      - **Manual installation** (global python or some other manual setup)  
        `pip install -r requirement.txt`

## Usage
Add the node via `image` -> `WD14Tagger.pys`  
Models are automatically downloaded at runtime if missing.
- **model**: The interrogation model to use, the most popular is [wd-v1-4-convnextv2-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2).  
  Supports ratings, characters and general tags.
- **threshold**: The score for the tag to be considered valid
- **character_threshold**: The score for the character tag to be considered valid
- **exclude_tags** A comma separated list of tags that should not be included in the results

Quick interrogation of images is also available on any node that is displaying an image, e.g. a `LoadImage`, `SaveImage`, `PreviewImage` node.  
Simply right click on the node (or if displaying multiple images, on the image you want to interrogate) and select `WD14 Tagger` from the menu
Settings used for this are in the `settings` section of `pysssss.json`.

### Offline Use
Simplest way is to use it online, interrogate an image, and the model will be downloaded and cached, however if you want to manually download the models:
- Create a `models` folder (in same folder as the `wd14tagger.py`)
- Use URLs for models from the list in `pysssss.json`
- Download `model.onnx` and name it with the model name e.g. `wd-v1-4-convnext-tagger-v2.onnx`
- Download `selected_tags.csv` and name it with the model name e.g. `wd-v1-4-convnext-tagger-v2.csv`

## Requirements
`onnxruntime` (recommended, interrogation is still fast on CPU, included in requirements.txt)  
or `onnxruntime-gpu` (allows use of GPU, many people have issues with this, if you try I can't provide support for this)
