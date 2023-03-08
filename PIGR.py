import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.cluster import AffinityPropagation
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PIGR(nn.Module):

    def __init__(self, hidden_size, batch_size, side_info, abs, device, layer, neighbor):
        super(PIGR, self).__init__()
        self.n_items = side_info[0]
        self.max_len = side_info[1]
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.abs = abs
        self.layer = layer
        self.neighbor = neighbor
        self.emb = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0, max_norm = 1.5)
        self.pos_emb = nn.Embedding(self.max_len + 1, self.hidden_size,  padding_idx=0, max_norm = 1.5)
        self.wnoise = nn.Embedding(self.max_len + 1, self.hidden_size,  padding_idx=0, max_norm= 1.5)
        self.cluster_emb = nn.Embedding(self.max_len + 1, 1,  padding_idx=0, max_norm= 1)
        self.ffn1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.ffn2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.gru = nn.GRU(2 * self.hidden_size, self.hidden_size, 1)
        self.emb_dropout = nn.Dropout(0.3)
        self.sp = torch.nn.Softplus(threshold=1)

        self.ln = nn.LayerNorm(self.hidden_size)
        self.gate = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.bilinear = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=True)
        self.device = device

    def corrupted_feature(self, emb):
        corrupted_emb = emb[torch.randperm(emb.shape[0])]
        corrupted_emb = corrupted_emb[:, torch.randperm(corrupted_emb.shape[1])]
        return corrupted_emb

    def con_loss(self, e_c, e_s):
        negative_sample = self.corrupted_feature(e_c)
        pos = torch.sum(torch.mul(e_c, e_s), 1).view(-1)
        neg = torch.sum(torch.mul(negative_sample, e_s), 1).view(-1)
        one = torch.ones(pos.shape, dtype=torch.float, device=self.device)
        scores = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg))),
                           0)/self.batch_size
        return scores

    def compute_similarity(self, item_emb, wnoise, reversed_lengths):
        item_sim = torch.matmul(item_emb, torch.transpose(item_emb, 2, 1))
        item_norm = torch.norm(item_emb, p=2, dim=2, keepdim=True)
        norm = torch.matmul(item_norm, item_norm.permute(0,2,1)) + 1e-6
        item_sim = item_sim/norm
        diag = self.cluster_emb(reversed_lengths).squeeze()
        diag = torch.diag_embed(diag)
        item_sim = item_sim + diag
        wnoise = self.sp(torch.matmul(item_emb, wnoise.permute(0,2,1)))
        coff = torch.normal(0, 1, size=(item_emb.shape[0], item_emb.shape[1], item_emb.shape[1])).to(self.device)
        wnoise = torch.mul(coff, wnoise)
        return item_sim + wnoise

    def gen_edges(self, similarity, neighbor, threshold):
        values, indices = torch.topk(similarity, neighbor, 2, largest=True)
        mask = torch.zeros(similarity.shape, dtype=torch.float).to(self.device)
        one = torch.ones(indices.shape, dtype=torch.float).to(self.device)
        mask = torch.scatter(mask, 2, indices, one)
        mask = mask.bool()
        below_threshold = (similarity.detach() > threshold)
        mask = mask & below_threshold
        edges = mask.float()
        return edges

    def forward(self, seq, lengths, reverse_lengths):
        mask = torch.where(lengths > 0, 1, 0)
        hidden = self.emb(seq)
        if not self.abs:
            pos = self.pos_emb(reverse_lengths)
            hidden = torch.cat([hidden, pos], dim=-1)
            seq_view = torch.sum(self.bilinear(hidden), dim=1)/torch.sum(mask, dim=-1, keepdim=True)
            hidden = self.emb_dropout(hidden)

            gru_lengths = torch.sum(mask, dim=1).cpu()
            pre_hidden = pack_padded_sequence(hidden.permute(1,0,2), gru_lengths)
            gru_out, hidden = self.gru(pre_hidden)
            gru_out, lengths = pad_packed_sequence(gru_out)
            hidden = gru_out.permute(1, 0, 2)

        e_s = hidden[torch.arange(mask.shape[0]).long(), (torch.sum(mask, 1) - 1).long()]
        zero_layer = torch.sum(hidden, dim=1)/torch.sum(mask,dim=1, keepdim=True)
        layer_collect = [zero_layer]
        for i in range(self.layer):
            wnoise = self.wnoise(reverse_lengths)
            mask = (mask > 0).int()
            similarity = self.compute_similarity(hidden, wnoise, reverse_lengths)
            sim_mask = (torch.matmul(mask.unsqueeze(2).float(), mask.unsqueeze(1).float())) == 0.
            masked_similarity = similarity.masked_fill(sim_mask, -1e10)

            edges = self.gen_edges(masked_similarity, self.neighbor, -1e10)
            cluster_label = edges.permute(0,2,1)

            sim_softmax = torch.softmax(masked_similarity, dim=2)
            sim_softmax_mask = torch.mul(cluster_label, sim_softmax)
            update_hidden = torch.matmul(sim_softmax_mask, hidden)

            cluster_mask = torch.sum(sim_softmax_mask, dim=2)
            cluster_mask = F.normalize(cluster_mask, p=2, dim=1)
            if i > 0:
                FFN_mask = torch.sum(mask, dim=-1, keepdim=True)
                FFN_mask = (FFN_mask > 1.0).int().unsqueeze(2)
                update_hidden = torch.mul(FFN_mask, update_hidden)

            feed1 = torch.relu(self.ffn1(update_hidden))
            feed2 = self.ln(self.emb_dropout(self.ffn2(feed1)) + update_hidden)
            layer_collect.append(torch.sum(feed2 * cluster_mask.view(cluster_mask.shape[0], -1, 1).float(), 1))

        e_c = torch.stack(layer_collect, dim=1)
        e_c = torch.mean(e_c, dim=1)

        alpha = torch.sigmoid(self.gate(torch.cat([e_c,e_s], dim=-1)))
        sess_repre = self.emb_dropout(torch.mul(alpha, e_c) + torch.mul(1 - alpha, e_s))

        item_embs = (self.emb(torch.arange(self.n_items).to(self.device)))
        pre_scores = torch.matmul(sess_repre, (item_embs).permute(1, 0))
        con_loss = 0
        if not self.abs:
            con_loss = self.con_loss(sess_repre, seq_view)

        return pre_scores,  con_loss




