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
import folder_paths
from .pysssss import get_ext_dir, get_comfy_dir, download_to_file, update_node_status, wait_for_async, get_extension_config, log
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

config = get_extension_config()

defaults = {
    "model": "wd-v1-4-moat-tagger-v2",
    "threshold": 0.35,
    "character_threshold": 0.85,
    "replace_underscore": False,
    "trailing_comma": False,
    "exclude_tags": "",
    "ortProviders": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "HF_ENDPOINT": "https://huggingface.co"
}
defaults.update(config.get("settings", {}))

if "wd14_tagger" in folder_paths.folder_names_and_paths:
    models_dir = folder_paths.get_folder_paths("wd14_tagger")[0]
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
else:
    models_dir = get_ext_dir("models", mkdir=True)
known_models = list(config["models"].keys())

log("Available ORT providers: " + ", ".join(ort.get_available_providers()), "DEBUG", True)
log("Using ORT providers: " + ", ".join(defaults["ortProviders"]), "DEBUG", True)

def get_installed_models():
    models = filter(lambda x: x.endswith(".onnx"), os.listdir(models_dir))
    models = [m for m in models if os.path.exists(os.path.join(models_dir, os.path.splitext(m)[0] + ".csv"))]
    return models


class WD14Model:

    def __init__(self, model_name, replace_underscore=False):
        if model_name.endswith(".onnx"):
            model_name = model_name[0:-5]
        installed = list(get_installed_models())
        if not any(model_name + ".onnx" in s for s in installed):
            wait_for_async(lambda: download_model(model_name, None, None))

        name = os.path.join(models_dir, model_name + ".onnx")
        self.model = InferenceSession(name, providers=defaults["ortProviders"])

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
                if replace_underscore:
                    tags.append(row[1].replace("_", " "))
                else:
                    tags.append(row[1])
        self.tags = tags
        self.general_index = general_index
        self.character_index = character_index


def tag(image, model: WD14Model, threshold=0.35, character_threshold=0.85, exclude_tags="", trailing_comma=False, client_id=None, node=None):
    input = model.model.get_inputs()[0]
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

    label_name = model.model.get_outputs()[0].name
    probs = model.model.run([label_name], {input.name: image})[0]

    result = list(zip(model.tags, probs[0]))

    # rating = max(result[:general_index], key=lambda x: x[1])
    general = [item for item in result[model.general_index:model.character_index] if item[1] > threshold]
    character = [item for item in result[model.character_index:] if item[1] > character_threshold]

    all = character + general
    remove = [s.strip() for s in exclude_tags.lower().split(",")]
    all = [tag for tag in all if tag[0] not in remove]

    res = ("" if trailing_comma else ", ").join((item[0].replace("(", "\\(").replace(")", "\\)") + (", " if trailing_comma else "") for item in all))

    print(res)
    return res


async def download_model(model, client_id, node):
    hf_endpoint = os.getenv("HF_ENDPOINT", defaults["HF_ENDPOINT"])
    if not hf_endpoint.startswith("https://"):
        hf_endpoint = f"https://{hf_endpoint}"
    if hf_endpoint.endswith("/"):
        hf_endpoint = hf_endpoint.rstrip("/")

    url = config["models"][model]
    # https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/model.onnx
    url = url.replace("{HF_ENDPOINT}", hf_endpoint)
    url = f"{url}/resolve/main/"
    async with aiohttp.ClientSession(loop=asyncio.get_event_loop()) as session:
        async def update_callback(perc):
            nonlocal client_id
            message = ""
            if perc < 100:
                message = f"Downloading {model}"
            update_node_status(client_id, node, message, perc)

        try:
            await download_to_file(
                f"{url}model.onnx", os.path.join(models_dir,f"{model}.onnx"), update_callback, session=session)
            await download_to_file(
                f"{url}selected_tags.csv", os.path.join(models_dir,f"{model}.csv"), update_callback, session=session)
        except aiohttp.client_exceptions.ClientConnectorError as err:
            log("Unable to download model. Download files manually or try using a HF mirror/proxy website by setting the environment variable HF_ENDPOINT=https://.....", "ERROR", True)
            raise

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
    default = defaults["model"] + ".onnx"
    model = WD14Model(default if default in models else models[0])

    return web.json_response(await tag(image, model, client_id=request.rel_url.query.get("clientId", ""), node=request.rel_url.query.get("node", "")))


class WD14Tagger:
    @classmethod
    def INPUT_TYPES(s):
        extra = [name for name, _ in (os.path.splitext(m) for m in get_installed_models()) if name not in known_models]
        models = known_models + extra
        return {"required": {
            "image": ("IMAGE", ),
            "model": (models, { "default": defaults["model"] }),
            "threshold": ("FLOAT", {"default": defaults["threshold"], "min": 0.0, "max": 1, "step": 0.05}),
            "character_threshold": ("FLOAT", {"default": defaults["character_threshold"], "min": 0.0, "max": 1, "step": 0.05}),
            "replace_underscore": ("BOOLEAN", {"default": defaults["replace_underscore"]}),
            "trailing_comma": ("BOOLEAN", {"default": defaults["trailing_comma"]}),
            "exclude_tags": ("STRING", {"default": defaults["exclude_tags"]}),
        }}

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "tag"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def tag(self, image, model, threshold, character_threshold, exclude_tags="", replace_underscore=False, trailing_comma=False):
        tensor = image*255
        tensor = np.array(tensor, dtype=np.uint8)

        pbar = comfy.utils.ProgressBar(tensor.shape[0])
        tags = []
        wd14_model = WD14Model(model, replace_underscore)
        for i in range(tensor.shape[0]):
            image = Image.fromarray(tensor[i])
            tags.append(tag(image, wd14_model, threshold, character_threshold, exclude_tags, trailing_comma))
            pbar.update(1)
        return {"ui": {"tags": tags}, "result": (tags,)}


class WD14ModelLoader:

    @classmethod
    def INPUT_TYPES(s):
        extra = [name for name, _ in (os.path.splitext(m) for m in get_installed_models()) if name not in known_models]
        models = known_models + extra
        return {"required": {
            "model_name": (models, { "default": defaults["model"] }),
            "replace_underscore": ("BOOLEAN", {"default": defaults["replace_underscore"]}),
        }}

    RETURN_TYPES = ("WD14_MODEL", )
    RETURN_NAMES = ("model", )
    FUNCTION = "load"

    CATEGORY = "image"

    def load(self, model_name, replace_underscore):
        return (WD14Model(model_name, replace_underscore), )


class WD14TaggerOnly:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", ),
            "model": ("WD14_MODEL", ),
            "threshold": ("FLOAT", {"default": defaults["threshold"], "min": 0.0, "max": 1, "step": 0.05}),
            "character_threshold": ("FLOAT", {"default": defaults["character_threshold"], "min": 0.0, "max": 1, "step": 0.05}),
            "trailing_comma": ("BOOLEAN", {"default": defaults["trailing_comma"]}),
            "exclude_tags": ("STRING", {"default": defaults["exclude_tags"]}),
        }}

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "tag"

    CATEGORY = "image"

    def tag(self, image, model, threshold, character_threshold, exclude_tags="", trailing_comma=False):
        tensor = image*255
        tensor = np.array(tensor, dtype=np.uint8)

        pbar = comfy.utils.ProgressBar(tensor.shape[0])
        tags = []
        for i in range(tensor.shape[0]):
            image = Image.fromarray(tensor[i])
            tags.append(tag(image, model, threshold, character_threshold, exclude_tags, trailing_comma))
            pbar.update(1)
        return (tags, )

NODE_CLASS_MAPPINGS = {
    "WD14Tagger|pysssss": WD14Tagger,
    "WD14ModelLoader|pysssss": WD14ModelLoader,
    "WD14TaggerOnly|pysssss": WD14TaggerOnly,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WD14Tagger|pysssss": "WD14 Tagger 🐍",
}
