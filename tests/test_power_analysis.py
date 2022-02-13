#!/usr/bin/env python

"""Tests for `power_analysis` package."""
from enum import Enum, auto
from typing import Any, List

import pytest
import numpy as np

# DOESN'T WORK
# np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

# to avoid opening a window and manually closing it
# we are using https://github.com/jonathf/matplotlib-sixel
import matplotlib
matplotlib.use('module://sixel')

from power_analysis import interface
from power_analysis.analysis.arch import Instructions, LoopInstructions
from power_analysis.analysis.correlation import BaseCorrelation, HammingInputCorrelation
from power_analysis.analysis.display import Plot
from power_analysis.experiments import BaseExperiment, Capture

TEST_N_SAMPLE = 10


class TestInterface(interface.Interface):
    N_SAMPLES = TEST_N_SAMPLE

    def samples_for_instruction(self) -> int:
        return 3

    class CaptureEnum(Enum):
        ZERO = auto()
        PEAK = auto()
        RANDOM = auto()

    def firmware_disassembly(self):
        pass

    def target_read(self) -> bytes:
        return b'miao'

    def target_write(self, inp: bytes):
        self._last_input = inp

    def firmware_compile(self, *args, **kwargs):
        pass

    def firmware_upload(self, *args, **kwargs):
        pass

    def arm(self):
        pass

    def capture(self):
        entry = next(self.captures)

        if isinstance(entry, self.__class__.CaptureEnum):
            if entry == self.__class__.CaptureEnum.ZERO:
                return np.zeros(self.N_SAMPLES)
            elif entry == self.__class__.CaptureEnum.PEAK:
                v = np.zeros(self.N_SAMPLES)
                # the minus sign because of trace normalization wrt power consumption
                v[self.N_SAMPLES//2] = -BaseCorrelation.HW(self._last_input[0])

                return v
        elif isinstance(entry, np.ndarray):
            return entry
        elif isinstance(entry, Exception):
            raise entry

    def set_capture(self, captures):
        self.captures = iter(captures)

    def target_flush(self):
        pass

    def target_reset(self):
        pass

    def setup_interface(self):
        pass


@pytest.fixture
def experiment(tmp_path):
    """Create the experiment class and the environment needed to save a session"""
    class TestExperiment(TestInterface, BaseExperiment):
        PATH_SESSIONS = tmp_path

        def do_run(self, *args, count, **kwargs):
            for _ in range(count):
                inp = self.random_bytes(5)
                trace, output = self.single_capture(inp)
                self.results.append((inp, trace, output))

    return TestExperiment


@pytest.fixture
def capture_class(tmp_path):
    """In order to test load/save functionality we need to
    override PATH_SESSIONS with the temporary directory created
    for the tests."""
    class TestCapture(Capture):
        PATH_SESSIONS = tmp_path

    return TestCapture


@pytest.fixture
def capture(experiment):
    args = 'test', 'project'
    exp = experiment(*args)

    n = 400

    exp.set_capture([TestInterface.CaptureEnum.PEAK for _ in range(n)])

    return exp.run(count=n)


@pytest.fixture
def correlation(capture):
    return HammingInputCorrelation(capture, left=0, right=0)


def test_capture():
    length = 256
    inputs = [bytes([_]) for _ in range(length)]
    traces = [np.zeros(TEST_N_SAMPLE) for _ in range(length)]

    capture = Capture(inputs, traces)

    # capture.save('kebab')


def test_experiment(experiment, capture_class):
    """Test basic functionality."""

    assert experiment.PATH_SESSIONS

    args = 'test', 'project'
    exp = experiment(*args)

    exp.set_capture([TestInterface.CaptureEnum.ZERO for _ in range(100)])

    capture_first = exp.run(count=5)

    sessions = exp.sessions()
    assert "test.npy" in sessions
    assert len(sessions) == 1
    assert len(capture_first.traces) == 5

    # try to rerun it
    capture_second = exp.run(count=10)

    sessions = exp.sessions()
    assert "test.npy" in sessions
    assert len(sessions) == 1
    assert len(capture_second.traces) == 15

    # try to reinit and load only
    exp = experiment(*args)
    capture_third = exp.run(load_only=True)

    sessions = exp.sessions()
    assert "test.npy" in sessions
    assert len(sessions) == 1
    assert len(capture_third.traces) == 15

    capture_third.filter(slice(None, None, None))

    assert capture_third.inputs.shape == (15,)
    assert capture_third.traces.shape == (15, TEST_N_SAMPLE)
    assert capture_third.outputs.shape == (15,)

    capture_third.filter(slice(1, 5, None))

    assert capture_third.inputs.shape == (15,)
    assert capture_third.traces.shape == (15, 4)
    assert capture_third.outputs.shape == (15,)

    # see if Capture can load the session by itself
    capture_reload = capture_class(session='test')


def test_experiment_w_exception(experiment):
    """Test basic functionality."""

    class UnexpectedException(Exception):
        pass

    args = 'test', 'project'
    exp = experiment(*args)

    captures: List[Any] = [TestInterface.CaptureEnum.ZERO for _ in range(100)]
    captures.append(UnexpectedException("something wrong happened!!!! oh no!!!!!!!"))
    captures.extend(TestInterface.CaptureEnum.ZERO for _ in range(100))

    exp.set_capture(captures)

    with pytest.raises(UnexpectedException):
        capture_first = exp.run(count=500)

    sessions = exp.sessions()
    assert "test.npy" not in sessions
    assert len(sessions) == 0

    # try to rerun it
    capture_second = exp.run(count=10)

    sessions = exp.sessions()
    assert "test.npy" in sessions
    assert len(sessions) == 1
    assert len(capture_second.traces) == 110

    # try to reinit and load only
    exp = experiment(*args)
    capture_third = exp.run(load_only=True)

    sessions = exp.sessions()
    assert "test.npy" in sessions
    assert len(sessions) == 1
    assert len(capture_second.traces) == 110


def test_correlation(experiment):
    args = 'test', 'project'
    exp = experiment(*args)

    n = 40

    exp.set_capture([TestInterface.CaptureEnum.PEAK for _ in range(n)])

    capture = exp.run(count=n)

    correlation = HammingInputCorrelation(capture, left=0, right=0)
    pearson = correlation.pearson()

    assert pearson.shape == (TEST_N_SAMPLE, 2)

    # here we are looking for the "peak"
    assert np.argsort(pearson[:, 0])[0] == TEST_N_SAMPLE//2

    stats = correlation.statistics

    assert stats.shape == (len(correlation.classes), 4, TEST_N_SAMPLE)


def test_display(correlation):
    plot = Plot(correlation)
    print(plot.dataframe_from_pearsons())

    plot.pearson()


def test_display_w_threshold(correlation):
    plot = Plot(correlation)
    print(plot.dataframe_from_pearsons(threshold=.9))


def test_display_empty(correlation):
    plot = Plot(correlation)
    print(plot.dataframe_from_pearsons(threshold=1.1))


def test_display_matrix(capture):
    matrix = Plot.matrix(5, capture, HammingInputCorrelation)

    assert matrix.shape == (5*6/2, 4)

    Plot.matrix(5, capture, HammingInputCorrelation, only_diagonal=True)

    Plot.matrix(5, capture, HammingInputCorrelation, only_diagonal=True, key=b'\xde\xad\xc0\xde\x66')

    with pytest.raises(ValueError):
        Plot.matrix(5, capture, HammingInputCorrelation, only_diagonal=True, key=b'\xde\xad\xc0\xde')


def test_instructions():
    instrs_core = Instructions.build(
        """ 762:	d7 01       	movw	r26, r14
            764:	4d 91       	ld	r20, X+
            766:	7d 01       	movw	r14, r26
            768:	91 91       	ld	r25, Z+
            76a:	94 27       	eor	r25, r20
            76c:	89 2b       	or	r24, r25
            76e:	a2 17       	cp	r26, r18
            770:	b3 07       	cpc	r27, r19
            772:	b9 f7       	brne	.-18     	; 0x762 <main+0x9c>""", start=28, step=4)

    assert isinstance(instrs_core._instructions, list)

    assert instrs_core.step == 4
    assert instrs_core.start == 28
    assert len(instrs_core) == 9
    assert instrs_core.cycles == 12

    instrs_loop = instrs_core * 4

    assert len(instrs_loop) == 36
    assert instrs_loop.cycles == 48

    asdict = instrs_loop.asdict()

    assert isinstance(asdict, dict)

    assert len(asdict.keys()) == 48 * 4

    instrs_slice = instrs_core[1:3]

    assert len(instrs_slice) == 2
    assert type(instrs_slice) == Instructions
    assert instrs_slice.step == instrs_core.step
    assert instrs_slice.start == instrs_core.start + instrs_core.step


def test_instructions_loop_manual():
    """Check loop building functionality"""
    aes_roundkey_outerloop_head = Instructions.build("""
     84e:	90 93 05 06 	sts	0x0605, r25	; 0x800605 <__TEXT_REGION_LENGTH__+0x7de605>
     852:	60 91 36 23 	lds	r22, 0x2336	; 0x802336 <state>
     856:	70 91 37 23 	lds	r23, 0x2337	; 0x802337 <state+0x1>
     85a:	90 e1       	ldi	r25, 0x10	; 16
     85c:	89 9f       	mul	r24, r25
     85e:	e0 01       	movw	r28, r0
     860:	11 24       	eor	r1, r1
     862:	20 e0       	ldi	r18, 0x00	; 0
     864:	30 e0       	ldi	r19, 0x00	; 0
    """, single_cycle_for_first_instruction=True)

    aes_roundkey_innerloop_head = Instructions.build("""
     866:	f9 01       	movw	r30, r18
     868:	ec 0f       	add	r30, r28
     86a:	fd 1f       	adc	r31, r29
     86c:	ea 57       	subi	r30, 0x7A	; 122
     86e:	fd 4d       	sbci	r31, 0xDD	; 221
     870:	db 01       	movw	r26, r22
     872:	a2 0f       	add	r26, r18
     874:	b3 1f       	adc	r27, r19
     876:	90 e0       	ldi	r25, 0x00	; 0
    """)

    assert aes_roundkey_innerloop_head.cycles == 9

    aes_roundkey_innerloop_core = Instructions.build("""
     878:	41 91       	ld	r20, Z+
     87a:	8c 91       	ld	r24, X
     87c:	48 27       	eor	r20, r24
     87e:	4d 93       	st	X+, r20
     880:	9f 5f       	subi	r25, 0xFF	; 255
     882:	94 30       	cpi	r25, 0x04	; 4
     884:	c9 f7       	brne	.-14     	; 0x878 <AddRoundKey+0x30>
    """)

    assert aes_roundkey_innerloop_core.cycles == 10

    aes_roundkey_innerloop = Instructions.build_loop(
        aes_roundkey_innerloop_head,
        aes_roundkey_innerloop_core,
        iterations=4,
    )

    assert isinstance(aes_roundkey_innerloop._instructions, list)
    assert aes_roundkey_innerloop.single_cycle_for_first_instruction is False
    assert aes_roundkey_innerloop[len(aes_roundkey_innerloop_head)].mnemonic == aes_roundkey_innerloop_core[0].mnemonic
    assert aes_roundkey_innerloop[len(aes_roundkey_innerloop_head)] is not aes_roundkey_innerloop_core[0]

    print(aes_roundkey_innerloop)

    assert aes_roundkey_innerloop.cycles == \
           aes_roundkey_innerloop_head.cycles + 4 * aes_roundkey_innerloop_core.cycles - 1

    aes_roundkey_outerloop_core = aes_roundkey_innerloop + Instructions.build("""
     886:	2c 5f       	subi	r18, 0xFC	; 252
     888:	3f 4f       	sbci	r19, 0xFF	; 255
     88a:	20 31       	cpi	r18, 0x10	; 16
     88c:	31 05       	cpc	r19, r1
     88e:	59 f7       	brne	.-42     	; 0x866 <AddRoundKey+0x1e>
    """)

    assert aes_roundkey_outerloop_head.cycles == 14
    assert aes_roundkey_outerloop_core.cycles == aes_roundkey_innerloop.cycles + 6

    aes_roundkey_outerloop = Instructions.build_loop(
        aes_roundkey_outerloop_head,
        aes_roundkey_outerloop_core,
        iterations=4,
    )

    assert aes_roundkey_outerloop.cycles == \
           aes_roundkey_outerloop_head.cycles \
           + 4 * aes_roundkey_outerloop_core.cycles - 1
    assert aes_roundkey_outerloop.single_cycle_for_first_instruction is True


def test_loop_automated():
    asm = """
 7aa:	10 93 05 06 	sts	0x0605, r17	; 0x800605 <__TEXT_REGION_LENGTH__+0x7de605>
 7ae:	de 01       	movw	r26, r28
 7b0:	91 96       	adiw	r26, 0x21	; 33
 7b2:	fe 01       	movw	r30, r28
 7b4:	31 96       	adiw	r30, 0x01	; 1
 7b6:	9e 01       	movw	r18, r28
 7b8:	29 5d       	subi	r18, 0xD9	; 217
 7ba:	3f 4f       	sbci	r19, 0xFF	; 255
 7bc:	80 e0       	ldi	r24, 0x00	; 0
 7be:	4d 91       	ld	r20, X+
 7c0:	91 91       	ld	r25, Z+
 7c2:	49 13       	cpse	r20, r25
 7c4:	81 e0       	ldi	r24, 0x01	; 1
 7c6:	a2 17       	cp	r26, r18
 7c8:	b3 07       	cpc	r27, r19
 7ca:	c9 f7       	brne	.-14     	; 0x7be <main+0xa2>
 7cc:	88 23       	and	r24, r24
"""
    iterations = 3
    instrs = LoopInstructions.build(asm, iterations=iterations, step=12)

    assert instrs.header == 9
    assert instrs.body == 7
    assert len(instrs) == instrs.header + (iterations * instrs.body)
    assert instrs.step == 12
