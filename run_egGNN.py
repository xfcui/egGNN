#!/usr/bin/env -S python3 -Bu
# coding: utf-8
# @Auth: Jor<qhjiao@mail.sdu.edu.cn>
# @Date: Tue 07 Jun 2022 09:10:48 AM HKT
# @Desc: ligand protein binding affinity prediction

import os
import argparse

from egGNN.model import *
from egGNN.egGNN_pipeline import run_egGNN


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ligand', type=str, help='the ligand path (absolute)')
    parser.add_argument('--protein', type=str, help='the protein path (absolute)')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu num you use')

    args = parser.parse_args()
    return args


def run(args):
    affinity = run_egGNN(args)

    # delete
    with open('eggnn', 'a') as f:
        f.write(f"{args.ligand_name.split('_')[0]},{affinity}\n")


if __name__ == '__main__':
    args = get_parser()
    args.ligand_name = os.path.basename(args.ligand).split('.')[0]
    args.protein_name = os.path.basename(args.protein).split('.')[0]
    run(args)
