import abc
import os
import subprocess
import time
from abc import ABC
import numpy as np

from .utils import MixinLogger


def get_env(name: str) -> str:
    result = os.getenv(name)

    if result is None:
        raise ValueError(f"environment variable '{name}' is not set")

    return result


class Interface(abc.ABC):
    """Abstract class as skeleton to show what to implement in order to
    interact with the target (compiling, uploading and communicate with) and capture traces"""

    @abc.abstractmethod
    def setup_interface(self, *args, **kwargs) -> None:
        """initialize scope/target. If this fails should raise an exception."""

    @abc.abstractmethod
    def target_reset(self) -> None:
        """Reset the target"""

    @abc.abstractmethod
    def target_flush(self) -> None:
        """Flush the buffer between target and user"""

    @abc.abstractmethod
    def target_read(self) -> bytes:
        """Read some bytes from the target"""

    @abc.abstractmethod
    def target_write(self, inp: bytes) -> int:
        """Write some bytes to the target"""

    @abc.abstractmethod
    def firmware_compile(self, *args, **kwargs):
        """Compile a given firmware"""

    @abc.abstractmethod
    def firmware_disassembly(self, path_fw, name_function, *args, **kwargs):
        """"""

    @abc.abstractmethod
    def firmware_upload(self, *args, **kwargs):
        """Upload the firmware to the target"""

    @abc.abstractmethod
    def arm(self):
        """Indicate we are ready to capture a trace"""

    @abc.abstractmethod
    def capture(self):
        """Give me a trace"""

    @abc.abstractmethod
    def samples_for_instruction(self) -> int:
        """Indicate how many samples for instruction. Return None if it's not possible to know."""


class FailedCapture(Exception):
    pass


# FIXME: if the CW has not the firmware uploaded (red light on)
#        the capture doesn't fail!
class CWLiteInterface(MixinLogger, Interface):

    SCOPETYPE = 'OPENADC'
    PLATFORM = 'CWLITEXMEGA'

    SCOPE = None

    N_SAMPLES = 300

    @property
    def scope(self):
        return self.__class__.SCOPE

    @scope.setter
    def scope(self, scope):
        self.__class__.SCOPE = scope

    def init(self):
        import chipwhisperer as cw
        self.cw = cw
        # retrieve at soon as possible so that we can fail as soon as possible
        self.PATH_CW = get_env('PATH_CW')

        # FIXME: move cw and scope to class level maybe
        # so to be like a singleton
        #if hasattr(self.scope, 'connectStatus')
        try:
            if not self.scope.connectStatus:
                self.scope.con()
        except (NameError, AttributeError):
            self.scope = cw.scope()

        self.prog = cw.programmers.XMEGAProgrammer

        try:
            self.target = cw.target(self.scope)
        except IOError:
            print("INFO: Caught exception on reconnecting to target - attempting to reconnect to scope first.")
            print(
                "INFO: This is a work-around when USB has died without Python knowing. Ignore errors above this line.")
            self.scope = cw.scope()
            self.target = cw.target(self.scope)

        print("INFO: Found ChipWhisperer😍")
        print(self.scope)

    def firmware_path(self, project, extension):
        """Utility method to return the firmware path for this configuration"""
        return "{path_cw}/hardware/victims/firmware/{project}/{project}-{platform}{extension}".format(
            path_cw=self.PATH_CW,
            project=project,
            platform=self.PLATFORM,
            extension=extension,
        )

    def firmware_compile(self, project, options, *args, **kwargs):
        """This method copies a local project into the chipwhisperer's source tree
        and then compiles it using the indicated options."""
        p = subprocess.run(
            '''cp -rv {project} {path_cw}/hardware/victims/firmware/ && \
                cd {path_cw}/hardware/victims/firmware/{project} && \
                LANG=C make PLATFORM={platform} {options}'''.format(
                project=project, platform=self.PLATFORM, path_cw=self.PATH_CW, options=options,
            ),
            capture_output=True,
            shell=True)
        # print(p.stdout.decode('utf-8')[:-100])
        print(p.stderr.decode('utf-8'))
        p.check_returncode()

    def firmware_upload(self, path_fw, *args, **kwargs):
        self.cw.program_target(
            self.scope,
            self.prog,
            str(path_fw))

    def firmware_disassembly(self, path_elf, name_function, *args, **kwargs):
        p = subprocess.run(
            "avr-objdump  -D --disassemble-zeroes {fw} | sed -n '/<{name}>:/,/^$/p'".format(
                fw=path_elf, name=name_function),
            capture_output=True,
            shell=True)
        stdout = p.stdout.decode('utf-8')

        print(stdout)
        print(p.stderr.decode('utf-8'))

        return stdout

    def setup_interface(self):
        self.init()

        self.scope.default_setup()

        self.logger.info("set number of samples to %d", self.N_SAMPLES)
        self.scope.adc.samples = self.N_SAMPLES
        self.scope.adc.timeout = 5000

    def target_reset(self):
        self.scope.io.pdic = False
        time.sleep(0.1)
        self.scope.io.pdic = "high_z"
        time.sleep(0.1)

    def target_flush(self):
        # Flush garbage too
        self.target.flush()

    def target_write(self, inp: bytes) -> int:
        return self.target.write(inp)

    def target_read(self, *args, **kwargs) -> bytes:
        return self.target.read(*args, **kwargs)

    def arm(self):
        self.scope.arm()

    def capture(self):
        # scope.capture() -> True if capture timed out, false if it didn’t.
        is_captured = not self.scope.capture()

        if not is_captured:
            raise FailedCapture("CWLite failed to capture a trace")

        return self.scope.get_last_trace()

    def samples_for_instruction(self) -> int:
        return 4


class SiglentInterfaceMixin(Interface, MixinLogger, ABC):
    """This mixin provides with the method to interact ONLY with this scope.
    You need to realize the other methods."""
    MANAGER = None

    def __init__(self, *args, channel: int, **kwargs):
        super(SiglentInterfaceMixin, self).__init__()

        import pyvisa

        if self.MANAGER is not None:
            self.MANAGER.close()

        self.MANAGER = pyvisa.ResourceManager('@py')

        resources = self.MANAGER.list_resources()

        self.logger.info("Found the following resources")

        for resource in resources:
            self.logger.info(resource)

        self.channel = channel

    def connect(self, resource=None):
        resources = self.MANAGER.list_resources()

        if isinstance(resource, int) and resource < len(resources):
            resource = resources[resource]
        elif isinstance(resource, str) and resource in resources:
            pass
        else:
            raise ValueError(f"'{resource}' is not a valid index nor a valid resource identifier")

        self.logger.info("trying to connect to resource: '%s'" % resource)

        self.oscilloscope = self.MANAGER.open_resource(resource, write_termination='\n', query_delay=0.25)
        idn = self.oscilloscope.query('*IDN?')

        self.logger.info('Connected to device \'%s\'' % idn.strip("\n"))

    def setup(self, channel):
        response = self.oscilloscope.write("TRIG_DELAY 0US")

        sample_rate = self.oscilloscope.query('SANU C%d?' % channel)

        self.sample_rate = int(sample_rate[len('SANU '):-2])

        self.logger.info('detected sample rate of %d' % self.sample_rate)

    def arm(self):
        response = self.oscilloscope.query("INR?")
        self.logger.debug(f"INR? -> {response}")

        response = self.oscilloscope.query("INR?")
        self.logger.debug(f"INR? -> {response}")

        self.oscilloscope.write("ARM")

    def capture(self):

        while True:
            response = self.oscilloscope.query("INR?")
            value = int(response[4:])

            self.logger.debug(value)

            if (value & 1) != 0:
                break

            time.sleep(.1)

        # the response to this is binary data so we need to write() and then read_raw()
        # to avoid encode() call and relative UnicodeError
        self.oscilloscope.write('C%d: WF? DAT2' % (self.channel,))

        response = self.oscilloscope.read_raw()

        if not response.startswith(b'C%d:WF ALL' % self.channel):
            raise ValueError('error: bad waveform detected -> \'%s\'' % repr(response[:80]))

        index = response.index(b'#9')
        index_start_data = index + 2 + 9
        data_size = int(response[index + 2:index_start_data])
        # the reponse terminates with the sequence '\n\n\x00' so
        # is a bit longer that the header + data
        data = response[index_start_data:index_start_data + data_size]
        self.logger.debug('data size: %d' % data_size)

        # FIXME: FOR NOW IT'S SAVED ONLY HALF OF THE WAVEFORM
        #        AFTER THE TRIGGER
        half = len(data) // 2

        import struct
        # this is necessary to have signed conversion
        vector = np.array([struct.unpack("b", bytes([_]))[0] for _ in data[half:]])

        return vector
