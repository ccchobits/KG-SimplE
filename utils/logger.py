import os
import time


class Logger:
    def __init__(self, configs):
        self.configs = configs
        self.path = "../scripts/asset/log/"

    # log: .type: pandas.DataFrame
    def write(self, log):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(os.path.join(self.path, current_time + ".log"), "w") as f:
            f.write(log.to_string())
