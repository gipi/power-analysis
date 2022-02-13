import abc
from typing import Tuple
import logging
import functools
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm.auto import trange

from .arch import Instructions
from ..experiments import Capture


def create_dataframe(pearsons, instructions: Instructions = None, threshold=None):
    """Create a dataframe from the np.array containing the pearson correlation
    obtained from a XCorrelation (with shape (128, #samples)) and also sort each
    column separately."""
    df = pd.DataFrame.from_records(pearsons)

    # if the instructions are passed as used to customize the column names
    if instructions:
        # TODO: reorganize code around iterating Instructions
        #       since it's not clear if you should have the iteration
        #       around cycles or single instructions
        # i = iter(instructions.asdict().values())
        _insts_dict = instructions.asdict()

        def _rename_column(_orig):
            if _orig not in _insts_dict:
                return _orig

            _instr = _insts_dict[_orig]

            _mnemonic = _instr.mnemonic if hasattr(_instr, 'mnemonic') else ""

            return f"{_orig}: {_mnemonic}"

        df.columns = df.columns.map(_rename_column)

    # here we build the element of the DataFrame associating the
    # tuple (byte, correlation) using the correlation to sort
    # the column itself
    for column in df:
        new_column = []
        for b, element in enumerate(df[column]):
            new_column.append((bytes([b]), element))

        # https://stackoverflow.com/questions/66905494/sort-values-with-key-to-sort-a-column-of-tuples-in-a-dataframe
        # we are sorting respect to the absolute value of the pearson coefficient
        # since the pearsons list has only 128 values
        new_column = pd.Series(new_column).sort_values(ascending=False, ignore_index=True,
                                                       key=lambda col: col.map(lambda x: abs(x[1])))

        if threshold is not None and (np.isnan(new_column[0][1]) or new_column[0][1] < threshold):
            df.pop(column)
            continue

        df[column] = new_column

    return df


class BaseCorrelation(abc.ABC):
    """Calculate correlation between traces and leaking model of the inputs
    (or a tranformation of it)."""

    def __init__(self, capture: Capture, logging_level='INFO'):
        self.capture = capture

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging_level)

        self.normalize = False

    @staticmethod
    @functools.lru_cache()
    def HW(n):
        return bin(n).count("1")

    @abc.abstractmethod
    def mapping(self, inputs):
        """This function maps the I/O to the leaking model (hamming weight/distance
        or whatever is correlated to the power consumption)."""

    @property
    @functools.lru_cache()  # FIXME: cache for instance and method to reset it
    def classification(self):  # or distinguisher?
        """Return the corresponding hamming weight vector of the data"""
        return np.array([self.mapping(_) for _ in self.inputs])

    @property
    def classes(self):
        return np.unique(self.classification)

    @property
    def traces(self):
        """Return the traces present in the capture eventually normalized if the `normalize`
        attribute is set."""
        if self.normalize:
            return self._do_normalize()

        return self.capture.traces

    @property
    def inputs(self):
        return self.capture.inputs

    def stats_from_traces(self):
        """Calculate mean and standard deviation for the captured traces """
        avg = np.array([np.mean(_) for _ in self.capture.traces])
        std = np.array([np.std(_) for _ in self.capture.traces])

        return avg, std

    def _do_normalize(self):
        traces = self.capture.traces
        avgs, stds = self.stats_from_traces()

        return np.array([(trace - avg) / std for avg, std, trace in zip(avgs, stds, traces)])

    @property
    # @functools.lru_cache()
    def statistics(self):
        """Generate an aggregate statistics by hamming weights"""
        # this cubersome expression only to maintain the np.array status with the right shape 8)
        return np.array([(np.mean(t, axis=0), np.std(t, axis=0), np.max(t, axis=0), np.min(t, axis=0)) for t in
                         [np.array([
                             trace for hw, trace in zip(self.classification, self.traces) if hw == _]) for _ in
                             self.classes]])

    @functools.lru_cache()
    def pearson(self, start=0, end=None, progress_bar=0) -> np.array:
        """Calculate the pearson correlation and related p-values between traces and
        leaking model.
        """
        hamming_input = self.classification
        # vector of shape (# traces, # samples)
        traces = self.traces

        # here we are going to generate a vertical slice of correlation
        rows, nsamples = traces.shape  # rows is the # of traces, columns the # of samples

        # here double check that the shapes make sense
        assert traces.shape[0] == hamming_input.shape[0]

        results = []

        for sample in trange(start, end or nsamples, disable=progress_bar == 0):
            vertical = [traces[row][sample] for row in range(rows)]

            pearson, p_value = pearsonr(hamming_input, vertical)

            results.append((pearson, p_value))

        return np.array(results)


class HammingInputCorrelation(BaseCorrelation):
    """Calculate correlations between input bytes and traces.

    Note that is able to calculate the correlation also with the Hamming
    distance between couples of input bytes.

    The default mapping uses the bytes representation, subclasses must encode
    properly."""

    def __init__(self, *args, left: int, right: int, key: bytes = b'\x00', **kwargs):
        """`left` and `right` indicate the operands of the XOR operation. If they are equal
        is calculate the correlation with the input itself instead.

        `key` instead is used as a further xor operand."""
        self.args = args
        self.left = left
        self.right = right
        self.key = key[0]

        super().__init__(*args, **kwargs)

    def input_decode(self, inp: bytes) -> bytes:
        return inp

    def input_left(self, inp: bytes) -> int:
        inp = self.input_decode(inp)
        return inp[self.left]

    def input_right(self, inp: bytes) -> int:
        inp = self.input_decode(inp)
        return inp[self.right]

    def mapping(self, inp: bytes) -> int:
        value = self.input_left(inp) ^ self.input_right(inp) if self.left != self.right else self.input_left(inp)

        return self.HW(value ^ self.key)

    @property
    @functools.lru_cache()
    def pearsons(self, progress_bar=0):
        """Return all the pearsons for all the first 128 keys.

        Note: the remaining can be generated knowing that p(k) = -p(~k).
        """
        correlations = []
        progress_bar_pearson = progress_bar >> 1
        # we use half of the available key space since
        # the bitwise not key has the same correlation
        # of the original key but with the opposite sign
        # (see paper "Optimal statistical power analysis")
        for key in trange(128, disable=progress_bar == 0):
            corr = self.__class__(*self.args, left=self.left, right=self.right, key=bytes([key]))
            correlations.append(corr.pearson(progress_bar=progress_bar_pearson))

        # we return the 128 elements, who is going to receive it must know
        # that the other 128 can be calculated
        # TODO: maybe return a specific datatype?
        return np.array(correlations)

    def pearsons_df(self, instructions=None, threshold=None):
        pearsons = self.pearsons
        # pearsones.shape = (128, #samples, 2)
        return create_dataframe(pearsons[:, :, 0], instructions=instructions, threshold=threshold)

    @classmethod
    def matrix(cls, *args, length, key: bytes = None, only_diagonal=False, **kwargs) -> np.array:
        """Return the entries of the upper triangle matrix with the corresponding correlations.
        In the diagonal are contained the inputs. Take all the argument of the constructor
        but `left` and `right`.

        If `only_diagonal` is True then only the diagonal is calculated and returned."""

        if key and len(key) < length:
            raise ValueError("`key` must have at least length equal to `length`")

        results = []
        for left in range(length):
            for right in range(left, length):
                if only_diagonal and left != right:
                    continue
                _kwargs = dict(kwargs)

                if only_diagonal and key:
                    _kwargs['key'] = bytes([key[left]])

                results.append((left, right, cls(*args, left=left, right=right, **_kwargs)))

        return np.array(results)
