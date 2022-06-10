import os


class Config:
    DATA_DIR = "data"
    L1_N_TRIALS = 100
    L2_N_TRIALS = 20

    @classmethod
    def filepath(cls, filename):
        return os.path.join(cls.DATA_DIR, filename)
