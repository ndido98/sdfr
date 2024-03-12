# Introduction

This is the dev-kit for the [Synthetic Data for Face Recognition](https://idiap.ch/challenge/sdfr/) (SDFR) Competition, in the scope of the 18th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2024).
This dev kit contains multiple useful files which can be both helpful and necessary for managing the submissions to the competition.
The dev-kit was mainly designed with Python + PyTorch in mind, but the competition does not prohibit the use of any other programming language or framework.


The dev-kit contains the following files:
* This README `README.md`
* The environment file `sdfr_env.yaml`.
* A reference iResNet-50 architecture `iresnet.py`.
* An example on how to export a pytorch model to ONNX `example_onnx_export.py`.
* A subset of images used to generate the sanitizer scores `sanitizer_samples.bin`
* A sanitizer script, which will generate sanitizer scores `model_sanitizer.py`.
* Information about face alignment. 

## Environment

In this archive, we provide with an environment file `sdfr_env.yaml`. The scripts provided here have been tested with this environment. There is no requirement to use this specific environment, but for the submissions, information useful for reproducibility is expected to be provided, which also includes information about the environment used.

This environment file can be used by running `conda env create -f sdfr_env.yaml`. You can then activate the environment using `conda activate sdfr`.

## iResNet-50

The `iresnet.py` file contains the reference iResNet-50 architecture for the task 1 of the competition (can also be used for task 2). We expect participants who use Python + PyTorch to use this reference architecture. If you use another programming language and/or framework, we hope you can provide with your best judgement and best effort to use as close of an equivalent as possible to this definition.

## ONNX Export

In `example_onnx_export.py`, we provide with an example on how instantiate the iResNet-50 and how to export the model in pytorch to ONNX with a simple unit-test to verify that the prediction for the raw and exported model is the same. Only the trained model is expected, in the ONNX format.

Once your model is exported to ONNX, it will be tested using data in a specific format (same for task1 and task2):
* `shape=[BATCH_SIZE, 3, 112, 112]`
* The channel order will be RGB (and not BGR!)
* `dtype=float32`
* `range=[0,1]`
* Unnormalized. If you want to normalize it, specify the normalization such as in the `example_onnx_export.py` and wrap the model with the normalization as documented.
* You may resize the samples to your liking in your wrapped model, by following a similar process as for the normalization, but the starting size is as documented here.

Additionally, the expected output format of the embeddings are:
* `shape=[BATCH_SIZE, 512]` for task1 and `shape=[BATCH_SIZE, EMB_SIZE]` for task2 (as you can use any architecture with any embedding size you wish)
* `dtype=float32`
* Your embeddings will be normalized in our evaluation.

## Model sanitizer

The `model_sanitizer.py` will generate a small subset of scores from the subset of images in the `sanitizer_img/` folder which will be helpful to you as a participant to see whether the score distribution of your model being tested in its ONNX format closely matches with your expectations and will also be required to be submitted together with the ONNX model for task 1 and 2, to verify that we are able to reproduce those same scores on our side, as a sanity check.

To run the sanitizer, you can execute:
```bash
python model_sanitizer.py --model_path </PATH/TO/MODEL> --task <MY_TASK> --device <MY_DEVICE> --device_id <DEV_ID> --batch_size <BATCH_SIZE> --sanitizer_bin_path <PATH_TO_BIN> --output_folder <OUTPUT_FOLDER>
```

Most should be self-explanatory. Here are the details:
* Replace `</PATH/TO/MODEL>` with the path to the ONNX model which you will submit.
* Replace `<MY_TASK>` with `task1` or `task2`.
* Replace `<MY_DEVICE>` with `cpu` or `cuda`, depending on which device you want to do the inferencing on.
* Replace `<DEV_ID>` with the device id number of your device (useful if you have multiple GPUs and want to specify which one to use. Use `nvidia-smi` to get the device id, it should be just above the fan percentage). If in doubt, `0` should be the first device and should work.
* Replace `<PATH_TO_BIN>` with the path to the `sanitizer_samples.bin` binary file from the dev-kit, which contains the images used to generate the scores. By default, the default value should work unless you've moved this file.
* Replace `<BATCH_SIZE>` with the number representing the batch size. A too large number may cause an OOM error on CUDA devices. In that case, reduce this value.
* Replace `<OUTPUT_FOLDER>` with the folder where you want to output the scores to. The filename of the scores will be: `task{1,2}_sanitizer_scores.txt`.

The model used in `</PATH/TO/MODEL>` and corresponding `task{1,2}_sanitizer_scores.txt` score file are to be submitted jointly, on the submission plateform, to their corresponding task.

## Face Alignment
In our evaluation process, we will use facial images that have been aligned and cropped to a resolution of 112x112 in RGB format. For alignment and cropping with five landmarks, we follow the same procedure as in [1,2,3].

As the evaluation datasets are already cropped, it is desirable that the training data is also cropped in the same way. Many face recognition training datasets are already distributed in this cropped format, so there is no need to crop them again. However, if you are generating new synthetic data, it is advisable to apply the same cropping technique to ensure uniformity with both the training and evaluation datasets.

For those interested in implementing this procedure, a sample script is available at the following GitHub repository: [AdaFace - Face Alignment Script](https://github.com/mk-minchul/AdaFace/blob/master/face_alignment/align.py). The script can be utilised as follows:

```python
path = 'path_to_the_image'
aligned_rgb_img = align.get_aligned_face(path)
```

This script uses the MTCNN model for face detection and landmark estimation, followed by a landmark-based alignment and cropping. However, users have the flexibility to substitute MTCNN with more advanced models to potentially enhance the alignment accuracy.

### References

1. Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. ArcFace: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

2. Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, and Yu Qiao. Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503, 2016.

3. Kim M, Jain AK, Liu X. Adaface: Quality adaptive margin for face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition 2022 (pp. 18750-18759).

