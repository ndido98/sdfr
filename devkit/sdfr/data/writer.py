# coding=utf-8
"""
  @file: writer.py
  @data: 17 November 2023
  @author: Christophe Ecabert
  @email: christophe.ecabert@idiap.ch
"""
from typing import Optional, Iterable
import os
import os.path as osp
import sys


def _get_user_input(prompt: str, choices: Iterable[str]) -> str:
    res = ''
    while res not in choices:
        res = input(prompt)
    return res

class ScoreWriter:
    """ Simple score writer """

    @classmethod
    def for_task(cls,
                 task: str,
                 folder: Optional[str] = None,
                 override: bool = False) -> 'ScoreWriter':
        """ Create score writer for specific task """
        filename = f'{task}_sanitizer_scores.txt'
        return cls(folder=folder, filename=filename, override=override)

    def __init__(self,
                 filename: str,
                 folder: Optional[str] = None,
                 override: bool = False):
        if filename is None:
            filename = 'sanitizer_scores.txt'
        if folder is None:
            folder = os.getcwd()
        elif not osp.exists(folder):
            os.makedirs(folder)
        self.override = override
        self.fname = osp.join(folder, filename)
        self.fobj = None

    def open(self):
        """ Open new file """
        if self.fobj is not None:
            self.close()
        # Add check if file already exists ?
        if not self.override and osp.exists(self.fname):
            # Ask user what to do
            m = (f'There is already a file named `{self.fname}`, do you want'
                  ' to overwrite it ? [yes, no]')
            ans = _get_user_input(prompt=m, choices=('yes', 'no'))
            if ans == 'no':
                sys.exit(0)
        # Proceed with possible overwrite
        self.fobj = open(self.fname, 'wt', encoding='utf-8')

    def close(self):
        """ Close file if open """
        if self.fobj is not None:
            self.fobj.close()
            self.fobj = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, scores: Iterable[float]):
        """ Dump score into text file """
        for score in scores:
            if self.fobj is None:
                raise RuntimeError('Writer is not open! Call open() or use '
                                   'context manager before writing.')
            self.fobj.write(f'{score}\n')
