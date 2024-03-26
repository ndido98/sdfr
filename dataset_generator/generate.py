import argparse
import asyncio
import base64
import os
import random
import subprocess
import time
import socket
from pathlib import Path

import aiohttp
import pandas as pd
import submitit
import requests
from tqdm.asyncio import tqdm
from pathvalidate import sanitize_filename


INITIAL_PORT = 20200
NEGATIVE_PROMPT = """
(deformed iris, deformed pupils, semi-realistic, CGI, 3D, render, sketch, cartoon, drawing, anime:1.4),
text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate,
morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation,
deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured,
gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers,
too many fingers, long neck, hair in front of the eyes, hat, (shadows), (three-quarter pose), (face in profile:1.1)
""".replace("\n", " ")
STEPS = 25
SAMPLER = "DPM++ SDE Karras"
MODEL = "Realistic_Vision_V5.1.safetensors"


def wait_for_port(port: int, timeout: int = 60, delay: int = 2) -> bool:
    tmax = time.time() + timeout
    while time.time() < tmax:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(("127.0.0.1", port))
            return True
        except:
            if delay > 0:
                time.sleep(delay)
    return False


async def process_prompts_for_server(
    prompts: list[(str, str)],
    output: Path,
    port: int,
    n_batches_per_identity: int,
    batch_size: int,
    pbar: tqdm,
    semaphore: asyncio.Semaphore,
) -> None:
    url = f"http://127.0.0.1:{port}"
    # Each server gets just one request at a time, to avoid overloading the GPU
    async with aiohttp.ClientSession() as session:
        async with semaphore:
            for identity, description in prompts:
                identity_path = output / sanitize_filename(identity)
                identity_path.mkdir(parents=True, exist_ok=True)
                tqdm.write(f"Generating images of {identity} on port {port}")
                for i in range(n_batches_per_identity):
                    tqdm.write(f"Generating batch {i} of {identity} on port {port}")
                    seed = random.randint(1, 999999999)
                    prompt = f"RAW photo of {identity}"
                    payload = {
                        "prompt": prompt,
                        "negative_prompt": NEGATIVE_PROMPT,
                        "steps": STEPS,
                        "seed": seed,
                        "batch_size": batch_size,
                        "sampler_index": SAMPLER
                    }
                    response = await session.post(url=f"{url}/sdapi/v1/txt2img", json=payload)
                    response.raise_for_status()
                    r = await response.json()
                    for j, image in enumerate(r["images"]):
                        image_id = i * n_batches_per_identity + j
                        path = identity_path / f"{image_id:04}_{seed + j}.png"
                        image_decoded = base64.b64decode(image)
                        with open(path, "wb") as f:
                            f.write(image_decoded)
                        pbar.update(1)


async def worker_event_loop(prompts: list[(str, str)], output: Path, servers_count: int, n_batches_per_identity: int, batch_size: int) -> None:
    # Divide the prompts among the servers
    prompts_per_server = [prompts[i::servers_count] for i in range(servers_count)]
    # Create a progress bar
    pbar = tqdm(total=len(prompts) * n_batches_per_identity * batch_size)
    tasks = []
    for i in range(servers_count):
        port = INITIAL_PORT + i
        task = asyncio.create_task(process_prompts_for_server(prompts_per_server[i], output, port, n_batches_per_identity, batch_size, pbar, asyncio.Semaphore(1)))
        tasks.append(task)
    await asyncio.gather(*tasks)
    pbar.close()


def worker_main(prompts: list[(str, str)], output: Path, servers_count: int, n_batches_per_identity: int, batch_size: int) -> None:
    for i in range(servers_count):
        # Spawn the server on port INITIAL_PORT + i
        port = INITIAL_PORT + i
        print(f"Spawning server on port {port}")
        subprocess.run([
            "singularity",
            "instance",
            "start",
            "--nv",
            "--writable-tmpfs",
            "stable-diffusion-webui/stable-diffusion-webui.sif",
            f"server{i}",
            "--port",
            str(INITIAL_PORT + i),
            "--skip-torch-cuda-test",
        ], env={
            **os.environ,
            "SINGULARITYENV_CUDA_VISIBLE_DEVICES": str(i),
        })
        print(f"Waiting for server on port {port}")
        ready = wait_for_port(port, timeout=300, delay=5)
        if not ready:
            raise RuntimeError(f"Server {port} is not ready")
        print(f"Port {port} is open, sleeping an extra minute to be sure the server is up and running")
        time.sleep(60)
        print(f"Setting options for server on port {port}")
        url = f"http://127.0.0.1:{port}"
        option_payload = {
            "sd_model_checkpoint": MODEL
        }
        try:
            response = requests.post(url=f"{url}/sdapi/v1/options", json=option_payload)
        except requests.exceptions.ConnectionError as ex:
            raise RuntimeError(f"Server on port {port} is not responding") from ex
        response.raise_for_status()
        print(f"Server on port {port} is ready")
    asyncio.run(worker_event_loop(prompts, output, servers_count, n_batches_per_identity, batch_size))


def main(args: argparse.Namespace) -> None:
    prompts = pd.read_csv(args.prompts)
    prompts = prompts[["itemLabel", "itemDesc"]].values.tolist()
    # Preprare the prompt slices for each node
    prompts_per_node = [prompts[i::args.n_nodes] for i in range(args.n_nodes)]
    # Create the output directory
    args.output.mkdir(parents=True, exist_ok=True)
    # Create the executor
    executor = submitit.AutoExecutor(folder=args.output / "logs")
    executor.update_parameters(
        timeout_min=60 * 24,  # 24 hours
        slurm_partition=args.partition,
        slurm_account=args.account,
        slurm_mem="64G",
        slurm_gres=f"gpu:{args.n_gpus_per_node}",
        slurm_cpus_per_task=4,
        slurm_comment="dataset_generator",
    )
    executor.map_array(
        worker_main,
        prompts_per_node,
        [args.output] * args.n_nodes,
        [args.n_gpus_per_node] * args.n_nodes,
        [args.n_batches_per_identity] * args.n_nodes,
        [args.batch_size] * args.n_nodes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n_nodes", type=int, default=1)
    parser.add_argument("--n_gpus_per_node", type=int, default=1)
    parser.add_argument("--n_batches_per_identity", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--partition", type=str)
    parser.add_argument("--account", type=str)

    args = parser.parse_args()

    main(args)