import logging
import os

import numpy as np


class MixinLogger:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel('INFO')


class MixinRecord(MixinLogger):
    """Save and restore traces. It's not enforced any particular format."""

    PATH_SESSIONS = "sessions"

    @staticmethod
    def slugify(label: str):
        return label.translate(str.maketrans("/ ;", "___"))

    @classmethod
    def sessions(cls):
        """Return the previous sessions"""
        if not os.path.exists(cls.PATH_SESSIONS):
            raise Exception(f"you must create the `{cls.PATH_SESSIONS}` directory")

        dirpath, dirnames, filenames = next(os.walk(cls.PATH_SESSIONS))
        return filenames

    def get_path(self, name):
        return '{}.npy'.format(os.path.join(self.PATH_SESSIONS, self.slugify(name)))

    def load(self, name):
        """Try to retrieve traces for this session from the filesystem."""
        path = self.get_path(name)

        if not os.path.exists(path):
            return None

        self.logger.info("loading trace found at '{}'".format(path))

        return np.load(path, allow_pickle=True)[()]

    def save(self, name, data):
        path = self.get_path(name)
        self.logger.info("saving data at '{}'".format(path))
        np.save(path, data)
