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