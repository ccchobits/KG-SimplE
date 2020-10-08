import os
import numpy as np

import pandas as pd


class Reader:
    def __init__(self, configs):
        self.configs = configs
        if configs.debug:
            print("start loading train/validate/test data ...", flush=True)
        # train_data: .type: np.array, .shape: (n_train, 3)
        self.n_train, self.train_data = self.get_triplets("train")
        self.n_valid, self.valid_data = self.get_triplets("valid")
        self.n_test, self.test_data = self.get_triplets("test")
        self.n_ent = self.get_num("entity")
        self.n_rel = self.get_num("relation")
        if configs.bern:
            self.stat = self.head_tail_ratio()
        self.indices = np.arange(self.n_train)
        if configs.debug:
            print("loaded n_train: %d, n_valid: %d, n_test: %d, n_ent: %d, n_rel: %d" % (
                self.n_train, self.n_valid, self.n_test, self.n_ent, self.n_rel), flush=True)

    def get_triplets(self, mode="train"):
        """
        :param mode: [train | valid | test]
        :return:
        - 1. .type: int
        - 2. .type: torch.tensor .shape: (n_triplet, 3) .location: cpu
        """
        file_name = os.path.join(self.configs.dataset_path, self.configs.dataset, mode + "2id.txt")
        with open(file_name) as file:
            lines = file.read().strip().split("\n")
            n_triplets = int(lines[0])
            data = np.empty((n_triplets, 3), dtype=np.long)
            for i in range(1, len(lines)):
                line = lines[i]
                data[i - 1] = np.array([int(ids) for ids in line.split(" ")])
            assert n_triplets == len(data), "number of triplets is not correct."
            return n_triplets, data

    def get_num(self, target):
        """
        :param target: [entity | relation]
        :return: int
        """
        return int(open(os.path.join(self.configs.dataset_path, self.configs.dataset, target + "2id.txt")).readline().strip())

    def head_tail_ratio(self):
        """
        :return:
        stat: .type: np.array .shape:(n_rel, 2)
        """
        stat = np.empty((self.n_rel, 2))
        train_data_for_stat = pd.DataFrame(self.train_data, columns=["head", "tail", "relation"])
        for relation in range(self.n_rel):
            head_count = len(
                train_data_for_stat[train_data_for_stat["relation"] == relation][["head"]].groupby(by=["head"]))
            tail_count = len(
                train_data_for_stat[train_data_for_stat["relation"] == relation][["tail"]].groupby(by=["tail"]))
            stat[relation] = np.array([head_count / (head_count + tail_count), tail_count / (head_count + tail_count)])
        return stat

    def shuffle(self):
        np.random.permutation(self.indices)

    def next_batch(self, start, end):
        size = end - start
        pos_samples = self.train_data[self.indices[start: end]]
        neg_samples = [self.get_neg_samples(pos_samples) for i in range(self.configs.neg_ratio)]
        triplets = np.concatenate([pos_samples] + neg_samples, axis=0)

        pos_labels = np.ones(shape=(size,))
        neg_labels = -np.ones(shape=(size * self.configs.neg_ratio,))
        labels = np.concatenate([pos_labels, neg_labels], axis=0)
        return triplets, labels

    def get_neg_samples(self, pos_samples):
        size = len(pos_samples)
        new_ent = np.random.randint(low=0, high=self.n_ent, size=(size,))
        if self.configs.bern:
            head_or_tail = np.empty(size)
            rand = np.random.random(size)
            for i in range(size):
                if rand[i] < self.stat[pos_samples[i][2]][0]:
                    head_or_tail[i] = 1
                else:
                    head_or_tail[i] = 0
        else:
            head_or_tail = np.random.randint(low=0, high=2, size=(size,))
        neg_samples = np.copy(pos_samples)
        for i in range(size):
            if head_or_tail[i] == 0:
                neg_samples[i][0] = new_ent[i]
            else:
                neg_samples[i][1] = new_ent[i]
        return neg_samples

    def get_all_triplets(self):
        all_triplets = set()
        for dataset in [self.train_data, self.valid_data, self.test_data]:
            for triplet in dataset:
                all_triplets.add(tuple(triplet.tolist()))
        return all_triplets
