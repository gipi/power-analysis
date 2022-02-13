"""
Modules that deals with instructions and cycles.

For now is limited to AVR architecture, the real deal would be
to find some library that is capable of symbolic execution like
angr or capstone but bad enough they don't support AVR :(.
"""
from __future__ import annotations

import copy
import numbers
import re
from typing import Collection, List, Iterator
from dataclasses import dataclass


@dataclass
class Instruction:
    mnemonic: str
    cycles: int
    length: int  # in bytes

    def is_branch(self) -> bool:
        return self.mnemonic.startswith("br")

    def get_offset(self) -> int:
        """Return the integer value of the offset; used for branch instructions."""

        match = re.match(r".*\.([-+]?\d{1,3})", self.mnemonic)

        if not match:
            raise AttributeError(f"the instruction `{self.mnemonic}` has not integer operand")

        return int(match.group(1))


class Instructions:
    """Wrap a set of instruction in order to deal with them.

    It's possible to access single instructions via indexing or returning
    a new Instructions instance via slicing."""
    PATTERNS = [
        # (<mnemonic>, <cycles>, <length>)
        ("nop", 1, 2),
        (r"adiw\sr\d{1,2}, 0x[0-9A-F]+", 2, 2),
        (r"sts\s0x[0-9a-f]+, r\d{1,2}", 2, 4),
        (r"st\s[X|Y|Z]\+, r\d{1,2}", 1, 2),
        (r"mov\sr\d{1,2}, r\d{1,2}", 1, 2),
        (r"movw\sr\d{1,2}, r\d{1,2}", 1, 2),
        (r"ld\sr\d{1,2}, [X|Y|Z]\+?", 2, 2),
        (r"ldd\sr\d{1,2}, [X|Y|Z]\+\d+", 3, 2),
        (r"ldi\sr\d{1,2}, 0x[0-9A-F]+", 1, 2),
        (r"lds\sr\d{1,2}, 0x[0-9A-F]+", 3, 2),
        (r"eor\sr\d{1,2}, r\d{1,2}", 1, 2),
        (r"adc\sr\d{1,2}, r\d{1,2}", 1, 2),
        (r"add\sr\d{1,2}, r\d{1,2}", 1, 2),
        (r"subi\sr\d{1,2}, 0x[0-9A-F]+", 1, 2),
        (r"sbci\sr\d{1,2}, 0x[0-9A-F]+", 1, 2),
        (r"mul\sr\d{1,2}, r\d{1,2}", 2, 2),
        (r"and\sr\d{1,2}, r\d{1,2}", 1, 2),
        (r"or\sr\d{1,2}, r\d{1,2}", 1, 2),
        (r"cp\sr\d{1,2}, r\d{1,2}", 1, 2),
        (r"cpi\sr\d{1,2}, 0x[0-9A-F]+", 1, 2),
        (r"cpc\sr\d{1,2}, r\d{1,2}", 1, 2),
        # for the  branch instructions we indicate the number of cycles
        # when the fallthrough fails
        (r"breq\s\.[-+]?\d{1,3}", 2, 2),
        (r"brne\s\.[-+]?\d{1,3}", 2, 2),
        (r"cpse\sr\d{1,2}, r\d{1,2}", 1, 2),  # FIXME
        # (r"pop"),
        (r"rjmp\s\.[-+]?\d{1,3}", 2, 2),
        (r"ijmp", 2, 2),
        (r"jmp\s0x[0-9A-Fa-f]+", 3, 4),
    ]

    def __init__(self, instructions: List[Instruction], start=0, step=None, single_cycle_for_first_instruction: bool = False):
        """Create instance with the given `instructions`. It's possible to set the `start` index
        and the number of cycle of samples for instruction (`step`).

        `single_cycle_for_first_instruction` instruct to limit the number of cycle for the first instruction
        for particular case where the instruction is the one executing in multiple cycles but start the capture
        at the last one."""
        self._instructions = instructions
        self.start = start
        self.step = step
        self.single_cycle_for_first_instruction = single_cycle_for_first_instruction

        if self.single_cycle_for_first_instruction:
            self._instructions[0].cycles = 1

    def __str__(self):
        inner = "\n".join(["\t%s" % str(ins) for ins in self._instructions])

        return f"[\n{inner}\n]"

    def __mul__(self, other: int):
        # here we need a little of thinking out of the box
        # otherwise it's using the same deep copy for all
        # the copies.
        instructions = []

        for _ in range(other):
            instructions.extend(copy.deepcopy(self._instructions))

        return Instructions(instructions, step=self.step, start=self.start)

    def __add__(self, other: Instructions):
        return Instructions(copy.deepcopy(self._instructions) + copy.deepcopy(other._instructions))

    # def __repr__(self):
    #    pass

    def __iter__(self) -> Iterator[Instruction]:
        return iter(self._instructions)

    def __getitem__(self, item):
        """Return a new Instructions instance if a slice is used as argument,
        attributes such as `start` and `step` are inherited by it.

        Return a single Instruction element when an index is used. """
        cls = type(self)
        if isinstance(item, slice):
            new = cls(self._instructions[item])
            new.step = self.step

            start = item.start if item.start else 0
            delta = start if self.step is None else start * self.step

            new.start = self.start + delta

            return new
        elif isinstance(item, numbers.Integral):
            return self._instructions[item]
        else:
            raise TypeError(f"{cls.__name__} indices must be integers")

    @property
    def cycles(self):
        return sum([i.cycles for i in self._instructions])

    def __len__(self):
        return len(self._instructions)

    def asdict(self):
        output = {}
        cycle = self.start or 0
        step = self.step or 1

        for inst in self._instructions:
            for i in range(inst.cycles * step):
                output[cycle] = inst
                cycle += 1

        return output

    @classmethod
    def build(cls, dump: str, **kwargs) -> Instructions:
        """Return an instance of Instructions from a dump of objdump."""
        _patterns = [(re.compile(f".*({pattern}).*"), *_) for pattern, *_ in cls.PATTERNS]

        output = []

        for line in dump.split("\n"):
            # we skip empty line or line composed only of spaces
            if not line or line.isspace():
                continue
            for pattern, *_args in _patterns:
                match = pattern.match(line)

                if match is not None:
                    output.append(Instruction(match.group(1).replace("\t", " "), *_args))
                    break
            else:
                raise ValueError(f"No Instruction decoded: `{line}`")

        return cls(output, **kwargs)

    @classmethod
    def build_loop(cls, head: Instructions, core: Instructions, iterations: int) -> LoopInstructions:
        # find the last instruction
        last = core[-1]

        # check is a branch instruction
        if not last.mnemonic.startswith("br"):
            raise ValueError(
                "In order to construct a loop from the given instructions I need a br<something> as a last one")

        # build the loop
        loop = core * iterations
        # change the number of cycle of the last branch
        # so that the flow is unchanged
        loop[-1].cycles = 1

        return LoopInstructions(
            head._instructions + loop._instructions,
            header=len(head),
            body=len(core),
            iterations=iterations,
            start=head.start,
            step=head.step,  # TODO: check coherence attribute `step` for head and loop
            single_cycle_for_first_instruction=head.single_cycle_for_first_instruction,
        )


# TODO: create a loop indicating from a dump which brX instruction
#       to use for determining the body (the start of the body can
#       be obtained from the offset of the brX opcode, this means
#       we need to add another field in the list of opcode with
#       the byte length)
class LoopInstructions(Instructions):
    """Create a loop of a certain number of iterations from a sample
    of instructions terminated by a branch instruction.

    The attribute `header` indicates the number of instructions of the prologue
    of the loop, the attribute `body` the number of instructions of the body of the loop.

    A diagram of a typical loop looks like the following

     [    head     ] [[ body ] [ body ] ...loop... [ body ]]
    """
    def __init__(self, *args,  header: int, body: int, iterations: int, **kwargs):
        super().__init__(*args, **kwargs)

        self.header = header
        self.body = body
        self.iterations = iterations

    @classmethod
    def build(cls, dump: str, iterations: int = 1, **kwargs) -> LoopInstructions:
        instrs = Instructions.build(dump, **kwargs)

        # find the brX instruction
        index_br = None
        for index, instr in enumerate(instrs):
            if instr.is_branch():
                index_br = index
                break
        else:
            raise ValueError("there is not branch instruction to create a loop from")

        instr_branch = instrs[index_br]

        # now we have to navigate backward the list to get where the branch would land
        offset = instr_branch.get_offset() + 2  # AVR pipelining need this extra offset
        direction = 1 if offset > 0 else -1  # in theory we should look only at negative offset since is a loop

        if offset >= 0:
            raise ValueError("there is an infinite loop or branching forward")

        index_body = index_br

        while offset != 0:  # there is the risk of an infinite loop
            instr = instrs[index_body]
            offset = offset - (direction * instr.length)
            index_body = index_body + direction

        # now instr point to the landing instruction that is the start
        # of the body of the loop
        instrs_body = instrs[index_body:index_br + 1]
        instrs_head = instrs[:index_body]

        return super().build_loop(instrs_head, instrs_body, iterations)

