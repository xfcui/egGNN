#! /usr/bin/env -S python3 -Bu
# coding: utf-8
# @Auth: Jor<qhjiao@mail.sdu.edu.cn>
# @Date: Sat 03 Jul 2021 06:59:36 PM HKT
# @Desc: construct graph for the molecule

import os
import dgl
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from dgllife.utils import mol_to_bigraph

CWD = os.path.dirname(os.path.abspath(__file__))


class ConstructGraph():

    def __init__(self):
        pass

    def one_dog_encoder(self, x, one_hot):
        return one_hot[x].values if x in one_hot else one_hot['Unknown'].values
        
    def bondf(self, mol):
        feats = []
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        bond_types_one_hot = pd.get_dummies([0, 1, 2, 3])
        bool_one_hot = pd.get_dummies([True, False])
        
        for bond in mol.GetBonds():
            bond_feat = []
            bond_type = bond.GetBondType()
            bond_feat.extend(self.one_dog_encoder(bond_types.index(bond_type) if bond_type in bond_types else 3, bond_types_one_hot))
            bond_feat.extend(self.one_dog_encoder(bond.GetIsAromatic(), bool_one_hot))
            bond_feat.extend(self.one_dog_encoder(bond.GetIsConjugated(), bool_one_hot))
            bond_feat.extend(self.one_dog_encoder(bond.IsInRing(), bool_one_hot))
            feats.extend([bond_feat, bond_feat])
        return {'bond_feat': torch.tensor(feats).float()}

    def atomf(self, mol):
        feats = []
        symbol_one_hot = pd.get_dummies(['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                                         'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
                                         'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
                                         'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
        aromatic_one_hot = pd.get_dummies([True, False])
        for atom in mol.GetAtoms():
            atom_feat = []
            atom_feat.extend(self.one_dog_encoder(atom.GetSymbol(), symbol_one_hot))
            atom_feat.extend([
                atom.GetAtomicNum(),    # a number
                atom.GetDegree(),       # a number
                atom.GetMass(),
                atom.GetFormalCharge(),    # ????????????
                atom.GetExplicitValence(),    # ?????????????????????
                atom.GetImplicitValence(),    # ?????????????????????
                atom.GetNumRadicalElectrons(),    # ?????? 0
            ])
            atom_feat.extend(self.one_dog_encoder(atom.GetIsAromatic(), aromatic_one_hot))
            feats.append(atom_feat)
        return {'atom_feat': torch.tensor(feats).float()}

    def construct_ligand_graph(self, mol, add_self_loop=True, num_virtual_nodes=1):
        return mol_to_bigraph(mol,
                              canonical_atom_order = False,
                              node_featurizer=self.atomf,
                              edge_featurizer=self.bondf,
                              add_self_loop=add_self_loop,
                              num_virtual_nodes=num_virtual_nodes)

    def construct_ligand_edge_graph(self, mol, add_self_loop=True):
        """????????????mol????????????????????????"""
        g = dgl.graph(([], []))
        bonds = mol.GetBonds()
        # ????????????
        g.add_nodes(len(bonds))
        # ?????????
        src_list, dst_list = [], []
        for i in range(len(bonds) - 1):
            bond1 = bonds[i]
            bond1_nodes = [bond1.GetBeginAtomIdx(), bond1.GetEndAtomIdx()]
            for j in range(i + 1, len(bonds)):
                bond2 = bonds[j]
                if bond2.GetBeginAtomIdx() in bond1_nodes or bond2.GetEndAtomIdx() in bond1_nodes:
                    src_list.extend([i, j])
                    dst_list.extend([j, i])

        # self loop
        if add_self_loop:
            nodes = g.nodes().tolist()
            src_list.extend(nodes)
            dst_list.extend(nodes)
        g.add_edges(src_list, dst_list)

        # ????????????????????????????????????feature
        g.ndata.update(self.bondf(mol))
        return g

    def construct_pocket_graph(self, mol, coords, cutoff=9, add_self_loop=False):
        aaindex1 = pd.read_csv(os.path.join(CWD, 'static/aaindex1_encoded.csv'))
        aaindex2 = pd.read_csv(os.path.join(CWD, 'static/aaindex_edge_encoded.csv'))
        g = dgl.graph(([], []))
        valid_res = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                     'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
        res_simple = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
                      'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
                      'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

        # ?????????????????????????????????
        aa_mid_coords = []
        res_list = []
        temp_res_count = 0
        for ind, atom in enumerate(mol.GetAtoms()):
            res = atom.GetPDBResidueInfo()
            res_name = res.GetResidueName()
            if res_name not in valid_res:    # ?????? HETATM
                continue
            if len(res_list) and res_name == res_list[-1]:
                aa_mid_coords[-1] += coords[ind]
                temp_res_count += 1
            else:
                if len(res_list):
                    aa_mid_coords[-1] /= temp_res_count
                    temp_res_count = 0
                aa_mid_coords.append(coords[ind])
                res_list.append(res_name)
        # ??????????????????
        aa_mid_coords[-1] /= temp_res_count
        
        # add nodes
        g.add_nodes(len(res_list))

        # ?????????????????????????????? ???????????? ??????????????????
        src_list = []
        dst_list = []
        edge_feat = []
        for i in range(0, len(aa_mid_coords) - 1):
            for j in range(i + 1, len(aa_mid_coords)):
                coor1 = aa_mid_coords[i]
                coor2 = aa_mid_coords[j]
                dist = np.sqrt(np.sum(np.square(coor1 - coor2)))
                if dist <= cutoff:
                    src_list.extend([i, j])
                    dst_list.extend([j, i])
                    # ?????? aaindex ??????
                    _temp_feat1, _temp_feat2 = [dist], [dist]
                    res_sim_1 = res_simple[res_list[i]]
                    res_sim_2 = res_simple[res_list[j]]
                    if res_sim_1 + res_sim_2 in aaindex2:
                        _temp_feat1.extend(aaindex2[res_sim_1 + res_sim_2])
                        _temp_feat2.extend(aaindex2[res_sim_1 + res_sim_2])
                    elif res_sim_2 + res_sim_1 in aaindex2:
                        _temp_feat1.extend(aaindex2[res_sim_2 + res_sim_1])
                        _temp_feat2.extend(aaindex2[res_sim_2 + res_sim_1])
                    else:
                        print("error in get aaindex2")
                        return
                    edge_feat.extend([_temp_feat1, _temp_feat2])

        # self loop
        if add_self_loop:
            nodes = g.nodes().tolist()
            src_list.extend(nodes)
            dst_list.extend(nodes)
        g.add_edges(src_list, dst_list)

        # ?????????feature
        res_feat = []
        for res in res_list:
            res_feat.append(aaindex1[res])

        g.ndata['res_feat'] = torch.tensor(res_feat).float()
        g.edata['edge_feat'] = torch.tensor(edge_feat).float()
        return g