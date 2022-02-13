import abc
import sys
import logging
import random
from typing import Any, NoReturn

import numpy as np
# Automatically choose between `tqdm.notebook` and `tqdm.std`
from tqdm.autonotebook import trange

from .interface import Interface
from .utils import MixinRecord


class Alignment:
    @classmethod
    def align_couple(cls, traceA, traceB, width=20):
        """Return the offset the traces need to be aligned to. If is negative"""
        WINDOW_WIDTH = width

        deltasA = []
        deltasB = []

        def delta(template, background, width):
            deltas = []
            for idx in range(width):
                # and traceB as a "background"
                background_window = background[idx: idx + width]

                delta = np.sum((background_window - template) ** 2)

                deltas.append(delta)

            return np.array(deltas)

        # first we use traceA as a template
        deltasA = delta(traceA[:WINDOW_WIDTH], traceB, WINDOW_WIDTH)
        # here the reverse
        deltasB = delta(traceB[:WINDOW_WIDTH], traceA, WINDOW_WIDTH)

        minA = np.argsort(deltasA)[0]
        minB = np.argsort(deltasB)[0]

        is_A = deltasA[minA] < deltasB[minB]

        return minA if is_A else -minB

    @classmethod
    def align(cls, traces, width=20):
        trace_as_reference = traces[0]

        output = [trace_as_reference]

        for trace in traces[1:]:
            offset = cls.align_couple(trace_as_reference, trace, width=width)

            if offset < 0:
                trace = np.append(np.zeros(-offset), trace[:offset])
            else:
                trace = np.append(trace[offset:], np.zeros(offset))

            output.append(trace)

        return np.array(output)

    @classmethod
    def align_capture(cls, capture):
        """Capture is a session of shape (<# traces>, 3)."""
        traces_aligned = cls.align(capture[:, 1])
        return np.array([
            (r[0], aligned, r[2]) for r, aligned in zip(capture, traces_aligned)])


# TODO: add encode/decode
#       add metadata to the capture (like samples for instruction, disassembly and compilation options)
class Capture(MixinRecord):
    """Wrap the captured traces with methods to interact with them.

    Also set useful parameter like samples for instruction or the range
    to restrict further operation."""

    def __init__(self, inputs=None, traces=None, outputs=None, session: str = None, samples_for_instruction=None, align=False, window=100):
        super().__init__()

        self.session = session

        if inputs is None and traces is None and session is None:
            raise ValueError("You must indicate `session` or `inputs` and `traces`")
        elif session:
            results = self.load(session)

            if results is None:
                raise ValueError("No session with name `{}`".format(session))

            self.logger.info("loading {} elements".format(len(results)))

            inputs, traces, outputs = results[:, 0], results[:, 1], results[:, 2]

        self.align = align
        self.window = window

        # we need to unloop the traces to mantain the (# traces, # samples) shape
        traces = np.array([_ for _ in traces])

        self.rawinputs = inputs
        self.rawtraces = traces
        self.rawoutputs = outputs

        self.reset()

        self.samples_for_instruction = samples_for_instruction
        self.range = None

    def filter(self, range: slice, fn=None) -> NoReturn:
        """Filter the traces in the sample dimension by the range passed as argument.
        It should be a slice instance."""
        if fn:
            filtered = np.array(
                [(i, trace, o) for i, trace, o in zip(self.rawinputs, self.rawtraces, self.rawoutputs) if
                 fn(i, trace, o)])

            if not filtered:
                raise ValueError("you filtered too much, the result is empty!")

            inputs = filtered[:, 0]
            traces = np.array([_ for _ in filtered[:, 1]])
            outputs = filtered[:, 2]
        else:
            inputs = self.rawinputs
            traces = self.rawtraces
            outputs = self.rawoutputs

        self.range = range

        self.traces = traces[:, self.range]  # here we are filtering the samples axis
        self.inputs = inputs
        self.outputs = outputs

    def reset(self):
        self.inputs = self.rawinputs
        self.traces = self.rawtraces if not self.align else Alignment.align(self.rawtraces, width=self.window)
        self.outputs = self.rawoutputs


class Session(Interface, MixinRecord):
    PATH_SESSIONS = "sessions"
    """Encapsulate the data of a capture session.

    The sessions are saved into the directory `sessions/`.

    >>> s = Session('ldi')
    >>> s.capture(['NOP=1 LDI=%d' % ((1 << _) - 1) for _ in range(9)])
    """

    def __init__(self, name: str, options: dict):
        super().__init__()

        self.name = name
        self.options_hamming = options
        self.cache = None
        self.reset_inputs()

    def reset_inputs(self):
        # this is necessary to save already obtained traces
        # in case the board disconnects and we can restart
        # from the last
        self.results = []
        self.inputs = iter(self.options_hamming)
        self.input = next(self.inputs)

    @staticmethod
    def normalize_trace(self, trace: np.ndarray) -> np.ndarray:
        return -trace

    def do_captures_for_option(self, options, count=100):
        """TODO: rename this function to mirror the fact that this
        doesn't capture one trace but capture maintaining fixed
        the firmware on the device.

        NOTE: this flashes a new firmware, so keep in mind that by specification
        the flash memory after 10k cycles could lost reliability."""

        path_fw = ''
        self.logger.debug(options)
        self.firmware_compile(options)
        self.firmware_upload()
        disassembly = self.firmware_disassembly(path_fw, 'main')

        traces = []
        for _ in trange(count):
            trace = self.capture()

            if trace is None:
                self.logger.warning("trace not captured")
                continue

            traces.append(np.array((options, self.normalize_trace(trace), disassembly)))

        return np.array(traces)

    def capture(self, count: int = 100, load_only: bool = False) -> Capture:
        """Capture `count` traces and return a Capture instance.

        If  a session was already saved it's loaded as a starting list, if
        an exception stops the capture the routine should be able to resume
        from start of the last options group."""

        # we want the cache only to load **full completed** runs
        if self.cache is None:
            self.cache = self.load(self.name)

            if self.cache is not None:
                self.logger.info('using cache')
                self.results.extend(list(self.cache))

                if load_only:
                    results = np.array(self.results)
                    return Capture(results[:, 0], results[:, 1], results[:, 2])
        elif load_only:
            raise ValueError("you have an unfinished session, loading would cause lost of data")

        # we are going to simulate a "do {} while ()"
        while True:
            self.logger.info(f"capture for input {self.input}")
            self.results.extend(self.do_captures_for_option(self.input, count=count))

            try:
                self.input = next(self.inputs)
            except StopIteration:
                break

        # the final output is a tensor with shape (9*count, 3)
        self.results = np.array(self.results)

        self.save(self.name, self.results)

        # after saving the results we can reset the list so that rerunning
        # this method is not going to re-add the same traces from cache
        results = self.results
        self.reset_inputs()
        self.cache = None

        return Capture(results[:, 0], results[:, 1], results[:, 2])


# Implementation note: using an abstract class could seem too coupled as a design architecture
# BUT remember that these experiments ARE TIGHTLY COUPLED with the lab setup
# so passing around singleton that should be fixed anyway seems unnecessary to me
class BaseExperiment(MixinRecord, Interface, abc.ABC):
    """Class to wrap scope, target, saved captured traces and interaction via serial
    with the target."""

    FW_DIS_FUNCTIONS = []

    def __init__(self, name: str, project: str, options=None):
        super().__init__()
        self.name = name
        self.project = project
        self.options = options or ""
        self.cache = None

        self.setup_interface()  # this is like the __init__ for the Interface
        self.reset_results()

    def reset_results(self):
        # this is necessary to save already obtained traces
        # in case the board disconnects and we can restart
        # from the last
        self.results = []

    def _setup_target(self):
        self.firmware_compile()
        for name in self.FW_DIS_FUNCTIONS:
            self.firmware_disassembly(name)

        self.firmware_upload()

    @staticmethod
    def random_bytes(count) -> bytes:
        return bytearray(random.getrandbits(8) for _ in range(count))

    @staticmethod
    def calc_hamming_weight(n):
        return bin(n).count("1")

    @staticmethod
    def normalize_trace(trace):
        return -trace

    def single_capture(self, input_value: bytes, with_trace=True):
        # remember to reset
        self.target_flush()
        self.target_reset()  # TODO: Do we need to reset in the general case?
        # and then arm otherwise
        # the scope will trigger
        # during the reset itself
        self.arm()

        self.target_write(input_value)

        data = self.target_read()

        trace = self.capture()

        return self.normalize_trace(trace), data

    @abc.abstractmethod
    def do_run(self, **kwargs):
        """Implement the actual logic of the experiment. The attribute `results` must
        be used to store the triple (<input>, <trace>, <output>).

        The simplest implementation is something like this:

            def do_run(self, *args, count, **kwargs):
                for _ in range(count):
                    # remember to save
                    inp = self.random_bytes(5)
                    trace, output = self.single_capture(inp)
                    self.results.append((inp, trace, output)).
        """

    def run(self, load_only: bool = False, **kwargs):
        # we want the cache only to load **full completed** runs
        if self.cache is None:
            self.cache = self.load(self.name)

            if self.cache is not None:
                self.logger.info('using cache (loaded %d elements)' % len(self.cache))
                self.results.extend(list(self.cache))

                if load_only:
                    results = np.array(self.results)
                    return Capture(results[:, 0], results[:, 1], results[:, 2],
                                   samples_for_instruction=self.samples_for_instruction())
        elif load_only:
            raise ValueError("you have an unfinished session, loading would cause data loss")

        self.logger.info("-- Starting session")

        self.do_run(**kwargs)

        # the final output is a tensor with shape (9*count, 3)
        self.results = np.array(self.results)

        self.logger.info('saving %d elements' % len(self.results))
        self.save(self.name, self.results)

        # after saving the results we can reset the list so that rerunning
        # this method is not going to re-add the same traces from cache
        results = self.results
        self.reset_results()
        self.cache = None

        return Capture(results[:, 0], results[:, 1], results[:, 2], samples_for_instruction=self.samples_for_instruction())
