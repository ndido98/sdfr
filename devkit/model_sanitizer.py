#!/bin/env/python
# coding=utf-8
"""
  @file: model_sanitizer.py
  @data: 17 November 2023
  @author: Christophe Ecabert
  @email: christophe.ecabert@idiap.ch
"""
from typing import TYPE_CHECKING
from argparse import ArgumentParser
from tqdm import tqdm
from sdfr.model.inference import OnnxModel
from sdfr.data.datasets import VerificationDataset, NumpyBatcher
from sdfr.data.writer import ScoreWriter


if TYPE_CHECKING:
    from argparse import Namespace


def main(args: 'Namespace'):
    """ Sanitizer """

    # Sanitize Model
    model = OnnxModel(model_path=args.model_path,
                      device=args.device,
                      device_id=args.device_id)

    model.validate_input_shape(expected=('batch_size', 3, 112, 112),
                               dtype='tensor(float)')

    o_shape = (('batch_size', 512) if args.task == 'task1' else
               ('batch_size', None))
    model.validate_output_shape(expected=o_shape, dtype='tensor(float)')

    # Score Sanitization
    scores = []
    with ScoreWriter.for_task(task=args.task,
                              folder=args.output_folder) as writer:
        dset = VerificationDataset(args.sanitizer_bin_path)
        n_elem = len(dset)
        n_elem = (n_elem + args.batch_size - 1) // args.batch_size
        for im1, im2 in tqdm(NumpyBatcher(dset, size=args.batch_size),
                             total=n_elem):
            _scores = model.compute_scores(im1, im2)
            writer.write(scores=_scores)
            scores.extend(_scores.tolist())

if __name__ == '__main__':

    # Args
    p = ArgumentParser()

    p.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to onnx model to generate sanitizer scores for.')
    p.add_argument(
        '--task',
        type=str,
        required=True,
        choices=('task1', 'task2'),
        help='Which task apply to the model.')
    p.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=('cpu', 'cuda'),
        help='Device on which the computation is run. Default: cpu.')
    p.add_argument(
        '--device_id',
        type=int,
        default=0,
        help='Index of the device to run on. Default: 0.')
    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Number of images to process at once. Default: 32.')
    p.add_argument(
        '--sanitizer_bin_path',
        type=str,
        default='sanitizer_samples.bin',
        help='Path to binary file storing sanitizing samples. Default: sanitizer_samples.bin')
    p.add_argument(
        '--output_folder',
        type=str,
        help='Location where to place the generrated file, defaults to current'
             ' workdir')

    cmd_args = p.parse_args()
    main(cmd_args)
