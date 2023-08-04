# https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags

import comfy.utils
import asyncio
import aiohttp
import numpy as np
import csv
import os
import sys
import onnxruntime as ort
from onnxruntime import InferenceSession
from PIL import Image
from server import PromptServer
from aiohttp import web
from .pysssss import get_ext_dir, get_comfy_dir, download_to_file, update_node_status, wait_for_async, get_extension_config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


config = get_extension_config()

defaults = {
    "model": "wd-v1-4-moat-tagger-v2",
    "threshold": 0.35,
    "character_threshold": 0.85,
    "exclude_tags": ""
}
defaults.update(config.get("settings", {}))

models_dir = get_ext_dir("models", mkdir=True)
all_models = ("wd-v1-4-moat-tagger-v2", 
              "wd-v1-4-convnext-tagger-v2", "wd-v1-4-convnext-tagger",
              "wd-v1-4-convnextv2-tagger-v2", "wd-v1-4-vit-tagger-v2")


def get_installed_models():
    return filter(lambda x: x.endswith(".onnx"), os.listdir(models_dir))


async def tag(image, model_name, threshold=0.35, character_threshold=0.85, exclude_tags="", client_id=None, node=None):
    if model_name.endswith(".onnx"):
        model_name = model_name[0:-5]
    installed = list(get_installed_models())
    if not any(model_name + ".onnx" in s for s in installed):
        await download_model(model_name, client_id, node)

    name = os.path.join(models_dir, model_name + ".onnx")
    model = InferenceSession(name, providers=ort.get_available_providers())

    input = model.get_inputs()[0]
    height = input.shape[1]

    # Reduce to max size and pad with white
    ratio = float(height)/max(image.size)
    new_size = tuple([int(x*ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height-new_size[0])//2, (height-new_size[1])//2))

    image = np.array(square).astype(np.float32)
    image = image[:, :, ::-1]  # RGB -> BGR
    image = np.expand_dims(image, 0)

    # Read all tags from csv and locate start of each category
    tags = []
    general_index = None
    character_index = None
    with open(os.path.join(models_dir, model_name + ".csv")) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if general_index is None and row[2] == "0":
                general_index = reader.line_num - 2
            elif character_index is None and row[2] == "4":
                character_index = reader.line_num - 2
            tags.append(row[1])

    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input.name: image})[0]

    result = list(zip(tags, probs[0]))

    # rating = max(result[:general_index], key=lambda x: x[1])
    general = [item for item in result[general_index:character_index] if item[1] > threshold]
    character = [item for item in result[character_index:] if item[1] > character_threshold]

    all = character + general
    remove = [s.strip() for s in exclude_tags.lower().split(",")]
    all = [tag for tag in all if tag[0] not in remove]

    res = ", ".join((item[0].replace("(", "\\(").replace(")", "\\)") for item in all))

    print(res)
    return res


async def download_model(model, client_id, node):
    url = f"https://huggingface.co/SmilingWolf/{model}/resolve/main/"
    async with aiohttp.ClientSession(loop=asyncio.get_event_loop()) as session:
        async def update_callback(perc):
            nonlocal client_id
            message = ""
            if perc < 100:
                message = f"Downloading {model}"
            update_node_status(client_id, node, message, perc)

        await download_to_file(
            f"{url}model.onnx", os.path.join("models",f"{model}.onnx"), update_callback, session=session)
        await download_to_file(
            f"{url}selected_tags.csv", os.path.join("models",f"{model}.csv"), update_callback, session=session)

        update_node_status(client_id, node, None)

    return web.Response(status=200)


@PromptServer.instance.routes.get("/pysssss/wd14tagger/tag")
async def get_tags(request):
    if "filename" not in request.rel_url.query:
        return web.Response(status=404)

    type = request.query.get("type", "output")
    if type not in ["output", "input", "temp"]:
        return web.Response(status=400)

    target_dir = get_comfy_dir(type)
    image_path = os.path.abspath(os.path.join(
        target_dir, request.query.get("subfolder", ""), request.query["filename"]))
    c = os.path.commonpath((image_path, target_dir))
    if os.path.commonpath((image_path, target_dir)) != target_dir:
        return web.Response(status=403)

    if not os.path.isfile(image_path):
        return web.Response(status=404)

    image = Image.open(image_path)

    models = get_installed_models()
    model = next(models, defaults["model"])

    return web.json_response(await tag(image, model, client_id=request.rel_url.query.get("clientId", ""), node=request.rel_url.query.get("node", "")))


class WD14Tagger:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", ),
            "model": (all_models, ),
            "threshold": ("FLOAT", {"default": defaults["threshold"], "min": 0.0, "max": 1, "step": 0.05}),
            "character_threshold": ("FLOAT", {"default": defaults["character_threshold"], "min": 0.0, "max": 1, "step": 0.05}),
            "exclude_tags": ("STRING", {"default": defaults["exclude_tags"]}),
        }}

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "tag"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def tag(self, image, model, threshold, character_threshold, exclude_tags=""):
        tensor = image*255
        tensor = np.array(tensor, dtype=np.uint8)

        pbar = comfy.utils.ProgressBar(tensor.shape[0])
        tags = []
        for i in range(tensor.shape[0]):
            image = Image.fromarray(tensor[i])
            tags.append(wait_for_async(lambda: tag(image, model, threshold, character_threshold, exclude_tags)))
            pbar.update(1)
        return {"ui": {"tags": tags}, "result": (tags,)}


NODE_CLASS_MAPPINGS = {
    "WD14Tagger|pysssss": WD14Tagger,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WD14Tagger|pysssss": "WD14 Tagger 🐍",
}
