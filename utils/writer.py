import time


class Writer:
    def __init__(self, configs):
        self.configs = configs
        self.filtered_arguments = ["save_path", "seed", "dataset_path", "mode", "log", "gpu"]
        self.default_span = 5
        self.span = {"bern": 5, "bs": 5, "dataset": 5, "dim": 3, "epochs": 4, "init_lr": 4, "lr_decay": 3,
                     "margin": 3, "norm": 1, "model": 4, "loss": 6}
        self.head = ["model"]
        self.tail = ["hidden"]
        self.order = self.get_order()
        self.path = "../scripts/asset/performance.result"

    # performance.type: pd.DataFrame
    def write(self, performance):
        with open(self.path, "a") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "|")
            all_arguments = sorted(
                filter(lambda x: x[0] not in self.filtered_arguments, list(vars(self.configs).items())),
                key=lambda x: self.order[x[0]])
            for key, value in all_arguments:
                if key in self.span:
                    span = self.span[key]
                else:
                    span = self.default_span
                f.write(key + ":" + ("%-" + str(span) + "s") % value + "|")
            f.write("\n")
            f.write(performance.to_string() + "\n")

    def get_order(self):
        all_keys = list(vars(self.configs))
        orderd_keys = self.head + sorted(
            set(all_keys).difference(set(self.head + self.filtered_arguments + self.tail))) + self.tail
        order = dict()
        for i, key in enumerate(orderd_keys):
            order[key] = i
        return order
