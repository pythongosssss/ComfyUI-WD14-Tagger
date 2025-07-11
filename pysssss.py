import asyncio
import os
import json
import shutil
import inspect
import aiohttp
from server import PromptServer
from tqdm import tqdm

config = None


def is_logging_enabled():
    config = get_extension_config()
    if "logging" not in config:
        return False
    return config["logging"]


def log(message, type=None, always=False):
    if not always and not is_logging_enabled():
        return

    if type is not None:
        message = f"[{type}] {message}"

    name = get_extension_config()["name"]

    print(f"(pysssss:{name}) {message}")


def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def get_comfy_dir(subpath=None):
    dir = os.path.dirname(inspect.getfile(PromptServer))
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    return dir


def get_web_ext_dir():
    config = get_extension_config()
    name = config["name"]
    dir = get_comfy_dir("web/extensions/pysssss")
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir += "/" + name
    return dir


def get_extension_config(reload=False):
    global config
    if reload == False and config is not None:
        return config

    config_path = get_ext_dir("pysssss.user.json")
    if not os.path.exists(config_path):
        config_path = get_ext_dir("pysssss.json")

    if not os.path.exists(config_path):
        log("Missing pysssss.json and pysssss.user.json, this extension may not work correctly. Please reinstall the extension.",
            type="ERROR", always=True)
        print(f"Extension path: {get_ext_dir()}")
        return {"name": "Unknown", "version": -1}
    with open(config_path, "r") as f:
        config = json.loads(f.read())
    return config

def link_js(src, dst):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.name == "nt":
        try:
            import _winapi
            _winapi.CreateJunction(src, dst)
            return True
        except:
            pass
    try:
        os.symlink(src, dst)
        return True
    except:
        import logging
        logging.exception('')
        return False


def is_junction(path):
    if os.name != "nt":
        return False
    try:
        return bool(os.readlink(path))
    except OSError:
        return False

def install_js():
    src_dir = get_ext_dir("web/js")
    if not os.path.exists(src_dir):
        log("No JS")
        return

    should_install = should_install_js()
    if should_install:
        log("it looks like you're running an old version of ComfyUI that requires manual setup of web files, it is recommended you update your installation.", "warning", True)
    dst_dir = get_web_ext_dir()
    linked = os.path.islink(dst_dir) or is_junction(dst_dir)
    if linked or os.path.exists(dst_dir):
        if linked:
            if should_install:
                log("JS already linked")
            else:
                os.unlink(dst_dir)
                log("JS unlinked, PromptServer will serve extension")
        elif not should_install:
            shutil.rmtree(dst_dir)
            log("JS deleted, PromptServer will serve extension")
        return
    
    if not should_install:
        log("JS skipped, PromptServer will serve extension")
        return
    
    if link_js(src_dir, dst_dir):
        log("JS linked")
        return

    log("Copying JS files")
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def should_install_js():
    return not hasattr(PromptServer.instance, "supports") or "custom_nodes_from_web" not in PromptServer.instance.supports


def init(check_imports):
    log("Init")

    if check_imports is not None:
        import importlib.util
        for imp in check_imports:
            spec = importlib.util.find_spec(imp)
            if spec is None:
                log(f"{imp} is required, please check requirements are installed.", type="ERROR", always=True)
                return False

    install_js()
    return True


async def download_to_file(url, destination, update_callback, is_ext_subpath=True, session=None):
    close_session = False
    if session is None:
        close_session = True
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        session = aiohttp.ClientSession(loop=loop)
    if is_ext_subpath:
        destination = get_ext_dir(destination)
    try:
        proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        print("proxy:", proxy)
        proxy_auth = None
        if proxy:
            proxy_auth = aiohttp.BasicAuth(os.getenv("PROXY_USER", ""), os.getenv("PROXY_PASS", ""))

        async with session.get(url, proxy=proxy, proxy_auth=proxy_auth) as response:            
            size = int(response.headers.get('content-length', 0)) or None

            with tqdm(
                unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1], total=size,
            ) as progressbar:
                with open(destination, mode='wb') as f:
                    perc = 0
                    async for chunk in response.content.iter_chunked(2048):
                        f.write(chunk)
                        progressbar.update(len(chunk))
                        if update_callback is not None and progressbar.total is not None and progressbar.total != 0:
                            last = perc
                            perc = round(progressbar.n / progressbar.total, 2)
                            if perc != last:
                                last = perc
                                await update_callback(perc)
    finally:
        if close_session and session is not None:
            await session.close()

def wait_for_async(async_fn):
    try:
        import concurrent.futures
        # Check if we're in a running event loop
        asyncio.get_running_loop()
        # We're in a running loop, so run the async function in a separate thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, async_fn())
            return future.result()  # This blocks until complete
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(async_fn())

def update_node_status(client_id, node, text, progress=None):
    if client_id is None:
        client_id = PromptServer.instance.client_id

    if client_id is None:
        return

    PromptServer.instance.send_sync("pysssss/update_status", {
        "node": node,
        "progress": progress,
        "text": text
    }, client_id)

async def update_node_status_async(client_id, node, text, progress=None):
    if client_id is None:
        client_id = PromptServer.instance.client_id

    if client_id is None:
        return

    await PromptServer.instance.send("pysssss/update_status", {
        "node": node,
        "progress": progress,
        "text": text
    }, client_id)
