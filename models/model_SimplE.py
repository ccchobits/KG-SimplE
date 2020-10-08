import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")


class SimplE(nn.Module):
    def __init__(self, n_ent, n_rel, depth, margin, reg, num_batch):
        super(SimplE, self).__init__()

        self.margin = margin
        self.depth = depth
        self.reg = reg
        self.num_batch = num_batch
        self.ent_head_embedding = nn.Embedding(n_ent, depth)
        self.ent_tail_embedding = nn.Embedding(n_ent, depth)
        self.rel_embedding = nn.Embedding(n_rel, depth)
        self.rel_inv_embedding = nn.Embedding(n_rel, depth)
        self.all_embeddings = [self.ent_head_embedding, self.ent_tail_embedding, self.rel_embedding, self.rel_inv_embedding]

    def initialize(self):
        nn.init.xavier_normal_(self.ent_head_embedding)
        nn.init.xavier_normal_(self.ent_tail_embedding)
        nn.init.xavier_normal_(self.rel_embedding)
        nn.init.xavier_normal_(self.rel_inv_embedding)

        # self.ent_embedding.weight.data = F.normalize(self.ent_embedding.weight.data, dim=1)
        self.rel_embedding.weight.data = F.normalize(self.rel_embedding.weight.data, dim=1)
        self.rel_inv_embedding.weight.data = F.normalize(self.rel_inv_embedding.weight.data, dim=1)

    def get_score(self, heads, tails, rels, clamp=True):
        # shape: (batch_size, depth)
        h1 = self.ent_head_embedding(heads)
        h2 = self.ent_head_embedding(tails)
        t1 = self.ent_tail_embedding(tails)
        t2 = self.ent_tail_embedding(heads)
        r1 = self.rel_embedding(rels)
        r2 = self.rel_inv_embedding(rels)

        score = (torch.sum(h1 * r1 * t1, dim=1) + torch.sum(h2 * r2 * t2, dim=1)) / 2.0
        if clamp:
            score = torch.clamp(score, -20, 20)

        # return shape: (batch_size,)
        return score

    def forward(self, x, labels):
        self.ent_head_embedding.weight.data = F.normalize(self.ent_head_embedding.weight.data, dim=1)
        self.ent_tail_embedding.weight.data = F.normalize(self.ent_tail_embedding.weight.data, dim=1)

        # shape: (batch_size,)
        heads, tails, rels = x[:, 0], x[:, 1], x[:, 2]
        scores = self.get_score(heads, tails, rels)

        return F.softplus(-labels * scores).mean() + self.reg * self.get_regularization()

    def get_regularization(self):
        penalty = 0
        for embed in self.all_embeddings:
            penalty += torch.sum(embed.weight ** 2)
        return penalty / self.num_batch

