# coding=utf-8
"""
  @file: inference.py
  @data: 17 November 2023
  @author: Christophe Ecabert
  @email: christophe.ecabert@idiap.ch
"""
from typing import Union, Iterable, List, Sequence, Optional
import onnxruntime as ort
import numpy as np


_supported_device = ('cpu', 'cuda')
_providers = {'cpu': 'CPUExecutionProvider',
              'cuda': 'CUDAExecutionProvider'}
_ort2np_dtype = {'tensor(float16)': np.half,
                 'tensor(float)': np.float32,
                 'tensor(double)': np.float64,
                 'tensor(int8)': np.int8,
                 'tensor(uint8)': np.uint8,
                 'tensor(int16)': np.int16,
                 'tensor(int32)': np.int32}


def _convert_onnx_shape(shape: Iterable[Union[str, int]],
                        batch_size: int,
                        batch_key: str = 'batch_size') -> List[int]:
    shp = []
    for elem in shape:
        if isinstance(elem, str):
            elem = batch_size if elem == batch_key else int(elem)
        shp.append(elem)
    return shp


def _normalize(array: np.ndarray,
               dim: int = -1,
               eps: float = 1e-8) -> np.ndarray:
    # Embeddings norm
    a_norm = np.linalg.norm(array, axis=dim, keepdims=True)
    a_norm = np.clip(a_norm, a_min=eps, a_max=None)
    # Normalize
    return array / a_norm


def _same_shape(shape: Sequence, target: Sequence) -> bool:
    if len(shape) != len(target):
        return False
    for shp, tgt in zip(shape, target):
        if tgt is not None and shp != tgt:
            return False
    return True


def _validate_shape(ort_nodes, target, dtype:str, stype: str):
    valid = True
    if len(ort_nodes) != 1:
        valid = False

    if valid:
        ort_nodes = ort_nodes[0]
        valid = (_same_shape(ort_nodes.shape, target) and
                    ort_nodes.type == dtype)

    if not valid:
        if not isinstance(ort_nodes, list):
            ort_nodes = [ort_nodes]
        v = [f'{o.shape}, {o.type}' for o in ort_nodes]
        raise ValueError(f'{stype.capitalize()} shape must be `{target}` with'
                         f' `{dtype}` dtype, got {v}')


class OnnxModel:
    """ Inference on Onnx Model """

    def __init__(self,
                 model_path: str,
                 device: str,
                 device_id: Optional[int] = None):
        # Sanity check
        if device not in _supported_device:
            supported = '`{}`'.format('`, `'.join(_supported_device))
            m = f'Unsupported device `{device}`, must be one of `{supported}`'
            raise ValueError(m)
        if device_id is None:
            device_id = 0
        self.model_path = model_path
        self.device = device
        self.device_id = device_id
        self._ort_session = None

    @property
    def session(self) -> ort.InferenceSession:
        """ Inference session """
        if self._ort_session is None:
            # Select device
            provider = _providers[self.device]
            # Create inference session for specific model/device
            self._ort_session = ort.InferenceSession(self.model_path,
                                                     providers=[provider])

        return self._ort_session

    def validate_input_shape(self,
                             expected: Iterable[Union[str, int]],
                             dtype: str):
        """ Validate input shape """
        _validate_shape(self.session.get_inputs(),
                        target=expected,
                        dtype=dtype,
                        stype='input')

    def validate_output_shape(self,
                              expected: Iterable[Union[str, int]],
                              dtype: str):
        """ Validate input shape """
        _validate_shape(self.session.get_outputs(),
                        target=expected,
                        dtype=dtype,
                        stype='output')


    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        # Sanity check
        model_inputs = self.session.get_inputs()
        if len(model_inputs) != 1:
            raise ValueError('Model must have a single input, got '
                             f'{len(model_inputs)} instead!')

        # Bind inputs
        bindings = self.session.io_binding()

        x_ort = ort.OrtValue.ortvalue_from_numpy(inputs,
                                                 device_type=self.device,
                                                 device_id=self.device_id)
        bindings.bind_input(name=model_inputs[0].name,
                            device_type=x_ort.device_name(),
                            device_id=self.device_id,
                            element_type=inputs.dtype,
                            shape=x_ort.shape(),
                            buffer_ptr=x_ort.data_ptr())

        # Bind outputs
        for out in self.session.get_outputs():
            assert 'tensor' in out.type, \
                'Support only model that produce tensor(s)'

            o_shp = _convert_onnx_shape(out.shape,
                                        batch_size=inputs.shape[0],
                                        batch_key='batch_size')
            bindings.bind_output(name=out.name,
                                 device_type=self.device,
                                 device_id=self.device_id,
                                 element_type=_ort2np_dtype[out.type],
                                 shape=o_shp)

        # Run inference
        self.session.run_with_iobinding(bindings)

        # Retrieve outputs
        outputs = bindings.get_outputs()
        outputs = outputs[0]
        return outputs.numpy()

    def compute_scores(self,
                       image1: np.ndarray,
                       image2: np.ndarray,
                       stype: str = 'cosine') -> np.ndarray:
        """ Compute score for a given images pair """
        # Extract embedding
        embed1 = self(image1)
        embed2 = self(image2)
        # Normalize
        e1n = _normalize(embed1, dim=-1)
        e2n = _normalize(embed2, dim=-1)
        # Scores
        if stype == 'cosine':
            scores = np.sum(e1n * e2n, axis=-1)
        elif stype == 'l2':
            scores = np.sum((e1n - e2n) ** 2.0, axis=-1)
            scores = 4.0 - scores
        return scores
