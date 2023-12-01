import random
import requests
import io
import base64
from PIL import Image
from os.path import join
import os
from tqdm import tqdm

# Settings
###########################################################
output_path = 'E:\\StableDiffusion\\'
input_file_with_names = 'dataset_generator\\all_humans.txt'
batch_size = 8
how_many_batches = 8
###########################################################

# Generation Stuff
###########################################################
url = "http://127.0.0.1:7860"
negative_prompt = "(deformed iris, deformed pupils, semi-realistic, CGI, 3D, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, hair in front of the eyes, hat, (shadows), (three-quarter pose), (face in profile:1.1)"
steps = 25
sempler = "DPM++ SDE Karras"
model = "Realistic_Vision_V5.1.safetensors"
option_payload = {
    "sd_model_checkpoint": model
}
###########################################################

with open(input_file_with_names, 'r', encoding="utf8") as f:
    names = f.readlines()
    names = [n.strip() for n in names]

response = requests.post(url=f'{url}/sdapi/v1/options', json=option_payload)

for name in tqdm(names[:500]):

    name = ''.join(name.split('|')[0])

    directory = join(output_path, name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # fil ein which put image name and the seed
    log_file = open(join(directory, 'log_generation.txt'), 'w')

    for number in range(how_many_batches):

        seed = random.randint(1, 999999999)

        feature = 'RAW photo of ' + name

        # print(feature)

        payload = {
            "prompt": feature,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "seed": seed,
            "batch_size": batch_size,
            "sampler_index": sempler
        }

        response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

        r = response.json()

        for i in range(len(r['images'])):
            image = Image.open(io.BytesIO(base64.b64decode(r['images'][i])))
            image.save(join(directory, f'{str(number).zfill(4)}{"_"}{str(i).zfill(4)}.png'))
            log_file.write(f'{str(number).zfill(4)}{"_"}{str(i).zfill(4)}.png' + ' ' + str(seed) + '\n')

    log_file.close()