#! /usr/bin/env -S python3 -Bu
# coding: utf-8
# @Auth: Jor<qhjiao@mail.sdu.edu.cn>
# @Date: Mon 06 Jun 2022 04:40:33 PM HKT
# @Desc: Models

import dgl
import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.BatchNorm1d(out_dim))
    
    def forward(self, x):
        return self.dense(x)
    

class EBlock(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, num_heads, out_feats):
        super().__init__()
        self._num_heads = num_heads
        self._out_feats = out_feats
        self.project_edge = nn.Sequential(
            nn.Linear(edge_in_feats, num_heads * out_feats, bias=False),
            nn.LayerNorm(num_heads * out_feats),
        )
        self.project_node = nn.Sequential(
            nn.Linear(node_in_feats, num_heads * out_feats, bias=False),
            nn.GELU(),
            nn.LayerNorm(num_heads * out_feats),
        )
        self.project_out = nn.Sequential(
            nn.Linear(num_heads * out_feats, out_feats, bias=False),
            nn.GELU(),
            nn.LayerNorm(out_feats),
        )

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            g.ndata['hv'] = self.project_node(node_feats).view(-1, self._num_heads, self._out_feats)
            
            g.edata['he'] = torch.exp(self.project_edge(edge_feats).view(-1, self._num_heads, self._out_feats))

            g.update_all(dgl.function.u_mul_e('hv', 'he', 'm'), dgl.function.sum('m', 'h'))
            fn = g.ndata['h'].reshape(g.ndata['h'].shape[0], -1)
            return self.project_out(fn)
        
        
class EGCN(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, hidden_feats, out_feats, n=3):
        super().__init__()
        self._n = n
        self.alphas = nn.ParameterList([nn.Parameter(torch.Tensor([0.0]), requires_grad=True) for _ in range(n)])
        self.beltas = nn.ParameterList([nn.Parameter(torch.Tensor([1.0]), requires_grad=True) for _ in range(n)])
        self.eblocks = nn.ModuleList([EBlock(node_in_feats, edge_in_feats, hidden_feats, out_feats) for _ in range(n)])
    
    def forward(self, g, node_feats, edge_feats):
        out = out0 = self.eblocks[0](g, node_feats, edge_feats)
        for i in range(1, self._n):
            out = out0 + self.alphas[i] * self.eblocks[i](g, out0 + self.beltas[i] * node_feats, edge_feats)
            out0 = out
        
        return out
        

class EGGNN(nn.Module):
    def __init__(self, lig_in_dim, lig_out_dim, pro_in_dim, pro_out_dim):
        super().__init__()
        # ligand
        self.lig_dense = DenseBlock(lig_in_dim, lig_out_dim)
#         self.lig_conv1 = EGCN(lig_out_dim, 10, 2, lig_out_dim, n=8)
        self.lig_conv1 = nn.ModuleList([EGCN(lig_out_dim, 10, 2, lig_out_dim, n=5) for _ in range(4)])

        # protein
        self.pro_dense = DenseBlock(pro_in_dim, pro_out_dim)
#         self.pro_conv1 = EGCN(pro_out_dim, 21, 2, pro_out_dim, n=8)
        self.pro_conv1 = nn.ModuleList([EGCN(pro_out_dim, 21, 2, pro_out_dim, n=5) for _ in range(4)])

        # cat -> fully connect
        self.dense = nn.Sequential(
            nn.Linear((lig_out_dim + pro_out_dim) * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 350),
            nn.BatchNorm1d(350),
            nn.ReLU(),
            nn.Linear(350, 1),
        )

    def forward(self, inputs, device):
        lig_graph, pro_graph = inputs
        lig_graph = lig_graph.to(device)
        pro_graph = pro_graph.to(device)
        lig_atom_feat = lig_graph.ndata['atom_feat']
        lig_bond_feat = lig_graph.edata['bond_feat']
        pro_res_feat = pro_graph.ndata['res_feat']
        pro_edge_feat = pro_graph.edata['edge_feat']
        
        # ligand
        lig_emb = self.lig_dense(lig_atom_feat)
#         lig_out = self.lig_conv1(lig_graph, lig_emb, lig_bond_feat)
        lig_out = [conv(lig_graph, lig_emb, lig_bond_feat) for conv in self.lig_conv1]
        lig_out = torch.cat(lig_out, 1)

        # protein
        pro_emb = self.pro_dense(pro_res_feat)
#         pro_out = self.pro_conv1(pro_graph, pro_emb, pro_edge_feat)
        pro_out = [conv(pro_graph, pro_emb, pro_edge_feat) for conv in self.pro_conv1]
        pro_out = torch.cat(pro_out, 1)
        
        lig_graph.ndata['lig_out'] = lig_out
        lig_out = dgl.sum_nodes(lig_graph, 'lig_out')
        pro_graph.ndata['pro_out'] = pro_out
        pro_out = dgl.sum_nodes(pro_graph, 'pro_out')
        
        h = torch.cat([lig_out, pro_out], dim=1)
        h = self.dense(h)
        return h