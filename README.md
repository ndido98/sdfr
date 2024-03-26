# BioLab SDFR Challenge submission

This repository contains the code for BioLab's submission to the SDFR Challenge.
Use the Conda environment provided in the `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate sdfr
```

Before training, it is essential to correctly align the images. To do so, run the following script:

```bash
python dataset_generator/align.py --dataset-path /path/to/dataset --output-path /path/to/output --batch-size 1 --device cuda:0 --image-size 112 --num-workers 8
```

Our employed training sets are:
* [IDiff-Face (Synthetic Uniform)](https://drive.google.com/drive/folders/1-V2MuYrEBsaFrqkQDpAwf1eQx3l9I-9r?usp=sharing)
* [DigiFace-1M](https://github.com/microsoft/DigiFace1M)

We internally tested our model against [LFW](http://vis-www.cs.umass.edu/lfw/lfw.tgz).

The experiment configuration files are:
* `experiments/synth_idiffface.yml` for task 1
* `experiments/synth_all_iresnet100.yml` for task 2

Moreover, change the LFW root path in `experiments/train.yml` and `experiments/test.yml` accordingly.
In that root there must be a file called `test_pairs.txt`, which can be found in the `experiments/` directory of this repository.

To train the model, run:

```bash
python main.py fit -c experiments/train.yml -c experiments/experiment_file_here.yml
```

To test the model, run:
    
```bash
python main.py test -c experiments/test.yml -c experiments/experiment_file_here.yml --trainer.logger.init_args.id wandb_run_id --ckpt_path /path/to/checkpoint
```

## Forbidden datasets

To generate our celebrities dataset with Stable Diffusion, run the following commands:

```bash	
cd dataset_generator
python query_all_occupations.py
```

This will download from Wikidata a .csv file containing the subjects that will be generated.

To generate the dataset, we use [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui),
packaged in a [Singularity](https://docs.sylabs.io/guides/latest/user-guide/) image to allow for multi-node multi-GPU processing in a SLURM environment.

First, build the Singularity image:

```bash
cd dataset_generator/stable-diffusion-webui
sudo singularity build stable-diffusion-webui.sif stable-diffusion-webui.def
```

Then, run the following command to generate the dataset:

```bash
cd dataset_generator
python generate.py --prompts all_humans.csv --output /your/dataset_dir --n_nodes 1 --n_gpus_per_node 1 --batch_size 64
```

After generating the dataset, extract the ArcFace features and filter it:

```bash
python extract_embeddings.py --dataset /your/dataset_dir --num-workers 8
python filter.py --dataset /your/dataset_dir --output /your/classes/file.txt --min-images-per-class 4 --distance-metric cosine --distance-threshold 0.597 --n-jobs -1
```
