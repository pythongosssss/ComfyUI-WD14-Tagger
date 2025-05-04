# https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags

import comfy.utils
import comfy.model_management
import asyncio
import aiohttp
import numpy as np
import csv
import os
import sys
import onnxruntime as ort
from onnxruntime import InferenceSession
from PIL import Image
import hashlib
from server import PromptServer
from aiohttp import web
import folder_paths
import torch
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


async def tag(batch, model_name, threshold=0.35, character_threshold=0.85, exclude_tags="", replace_underscore=True, trailing_comma=False, client_id=None, node=None):
    if model_name.endswith(".onnx"):
        model_name = model_name[0:-5]
    installed = list(get_installed_models())
    if not any(model_name + ".onnx" in s for s in installed):
        await download_model(model_name, client_id, node)

    # unloaded = comfy.model_management.free_memory(1e30, torch.device(torch.cuda.current_device()))
    # if unloaded is not None and len(unloaded) > 0:
    #     torch.cuda.empty_cache()
    #     torch.cuda.ipc_collect()
    unloaded = comfy.model_management.unload_all_models()
    print(f"Unloaded models: {unloaded}")

    name = os.path.join(models_dir, model_name + ".onnx")
    model = InferenceSession(name, providers=defaults["ortProviders"])

    input = model.get_inputs()[0]
    height = input.shape[1]

    for i in range(len(batch)):
        # Reduce to max size and pad with white
        ratio = float(height)/max(batch[i].size)
        new_size = tuple([int(x*ratio) for x in batch[i].size])
        batch[i] = batch[i].resize(new_size, Image.LANCZOS)
        square = Image.new("RGB", (height, height), (255, 255, 255))
        square.paste(batch[i], ((height-new_size[0])//2, (height-new_size[1])//2))

        batch[i] = np.array(square).astype(np.float32)
        batch[i] = batch[i][:, :, ::-1]  # RGB -> BGR
        batch[i] = np.expand_dims(batch[i], 0)

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
            if replace_underscore:
                tags.append(row[1].replace("_", " "))
            else:
                tags.append(row[1])

    # imgs = np.array([im for im in batch])

    probs = []
    label_name = model.get_outputs()[0].name
    for img in batch:
        probs.append(model.run([label_name], {input.name: img})[0])
    # probs = probs[: len(batch)]
    # probs = model.run([label_name], {input.name: imgs})[0]

    # print(probs)

    res = []

    for i in range(len(batch)):
        result = list(zip(tags, probs[i][0]))

        # rating = max(result[:general_index], key=lambda x: x[1])
        general = [item for item in result[general_index:character_index] if item[1] > threshold]
        character = [item for item in result[character_index:] if item[1] > character_threshold]

        all = character + general
        remove = [s.strip() for s in exclude_tags.lower().split(",")]
        all = [tag for tag in all if tag[0] not in remove]

        res.append(("" if trailing_comma else ", ").join((item[0].replace("(", "\\(").replace(")", "\\)") + (", " if trailing_comma else "") for item in all)))

    return res


async def download_model(model, client_id, node):
    hf_endpoint = os.getenv("HF_ENDPOINT", defaults["HF_ENDPOINT"])
    if not hf_endpoint.startswith("https://"):
        hf_endpoint = f"https://{hf_endpoint}"
    if hf_endpoint.endswith("/"):
        hf_endpoint = hf_endpoint.rstrip("/")

    url = config["models"][model]
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
    model = default if default in models else models[0]

    return web.json_response(await tag(image, model, client_id=request.rel_url.query.get("clientId", ""), node=request.rel_url.query.get("node", "")))


class WD14Tagger:
    def __init__(self):
        self.hash = {}              # settings hash --> list of tuples (hash of images, tags)
        self.max_cached = 100       # avoid oom

    def get_cache_size(self):
        items = 0
        for settings_hash in self.hash:
            items += len(self.hash[settings_hash])
        return items

    @classmethod
    def INPUT_TYPES(s):
        extra = [name for name, _ in (os.path.splitext(m) for m in get_installed_models()) if name not in known_models]
        models = known_models + extra
        return {
            "required": {
                "image": ("IMAGE", ),
                "model": (models, { "default": defaults["model"] }),
                "threshold": ("FLOAT", {"default": defaults["threshold"], "min": 0.0, "max": 1, "step": 0.05}),
                "character_threshold": ("FLOAT", {"default": defaults["character_threshold"], "min": 0.0, "max": 1, "step": 0.05}),
                "replace_underscore": ("BOOLEAN", {"default": defaults["replace_underscore"]}),
                "trailing_comma": ("BOOLEAN", {"default": defaults["trailing_comma"]}),
                "exclude_tags": ("STRING", {"default": defaults["exclude_tags"]}),
            },
            "optional": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 128}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "tag"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def tag(self, image, model, threshold, character_threshold, exclude_tags="", replace_underscore=False, trailing_comma=False, batch_size=1):

        if not isinstance(image, list):
            images = [image]
        else:
            images = image

        batches = []
        batch = []

        mem = comfy.model_management.get_total_memory(torch_total_too=True)
        total_vram = mem[0] / (1024 * 1024)
        total_vram_torch = mem[1] / (1024 * 1024)
        print("Total VRAM {:0.0f} MB, total Torch VRAM {:0.0f} MB".format(total_vram, total_vram_torch))

        # build hash for cache
        settings_hash = f'{len(model)}{hash(model)}-{threshold}-{character_threshold}-{len(exclude_tags)}{hash(exclude_tags)}-{replace_underscore}-{trailing_comma}-{batch_size}'
        img_hashes = []

        for image in images:
            tensor = image*255
            tensor = np.array(tensor.cpu(), dtype=np.uint8)

            for i in range(tensor.shape[0]):
                image = Image.fromarray(tensor[i])
                img_hashes.append(hashlib.md5(image.tobytes()).hexdigest())
                batch.append(image)
                if len(batch) == batch_size or i == tensor.shape[0] -1:
                    batches.append(batch)
                    batch = []

        img_hash = "-".join(img_hashes)

        # check cache for entry
        if settings_hash in self.hash:
            for stored_tags in self.hash[settings_hash]:
                if stored_tags[0] == img_hash:
                    print(f'hashed tags: {stored_tags[1]}')
                    return {"ui": {"tags": stored_tags[1]}, "result": (stored_tags[1],)}

        pbar = comfy.utils.ProgressBar(len(images))
        tags = []
        for batch in batches:
            tags = tags + wait_for_async(lambda: tag(batch, model, threshold, character_threshold, exclude_tags, replace_underscore, trailing_comma))
            pbar.update(len(batch))

        print(tags)

        # store tags in cache
        if settings_hash in self.hash:
            self.hash[settings_hash].insert(0, (img_hash, tags))
        else:
            self.hash[settings_hash] = [(img_hash, tags)]

        # prune cache to avoid oom
        while self.get_cache_size() > self.max_cached:
            # TODO: improve by using LRU mechanism
            for settings_hash in self.hash:
                if len(self.hash[settings_hash]) > 0: del self.hash[settings_hash][-1]
                if self.get_cache_size() <= self.max_cached: break

        mem = comfy.model_management.get_total_memory(torch_total_too=True)
        total_vram = mem[0] / (1024 * 1024)
        total_vram_torch = mem[1] / (1024 * 1024)
        print("Total VRAM {:0.0f} MB, total Torch VRAM {:0.0f} MB".format(total_vram, total_vram_torch))

        return {"ui": {"tags": tags}, "result": (tags,)}

NODE_CLASS_MAPPINGS = {
    "WD14Tagger|pysssss": WD14Tagger,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WD14Tagger|pysssss": "WD14 Tagger 🐍",
}
