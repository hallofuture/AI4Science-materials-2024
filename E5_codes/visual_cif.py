"""Visualizes CIF files, making a supercell."""
from pathlib import Path
import argparse
from tqdm import tqdm

from pymatgen.core import Structure
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vedo import settings, Spheres, Lines, Group, show, screenshot

settings.default_backend = '2d'

cmap = {
  "H": "FFFFFF",
  "He": "D9FFFF",
  "Li": "CC80FF",
  "Be": "C2FF00",
  "B": "FFB5B5",
  "C": "909090",
  "N": "3050F8",
  "O": "FF0D0D",
  "F": "90E050",
  "Ne": "B3E3F5",
  "Na": "AB5CF2",
  "Mg": "8AFF00",
  "Al": "BFA6A6",
  "Si": "F0C8A0",
  "P": "FF8000",
  "S": "FFFF30",
  "Cl": "1FF01F",
  "Ar": "80D1E3",
  "K": "8F40D4",
  "Ca": "3DFF00",
  "Sc": "E6E6E6",
  "Ti": "BFC2C7",
  "V": "A6A6AB",
  "Cr": "8A99C7",
  "Mn": "9C7AC7",
  "Fe": "E06633",
  "Co": "F090A0",
  "Ni": "50D050",
  "Cu": "C88033",
  "Zn": "7D80B0",
  "Ga": "C28F8F",
  "Ge": "668F8F",
  "As": "BD80E3",
  "Se": "FFA100",
  "Br": "A62929",
  "Kr": "5CB8D1",
  "Rb": "702EB0",
  "Sr": "00FF00",
  "Y": "94FFFF",
  "Zr": "94E0E0",
  "Nb": "73C2C9",
  "Mo": "54B5B5",
  "Tc": "3B9E9E",
  "Ru": "248F8F",
  "Rh": "0A7D8C",
  "Pd": "006985",
  "Ag": "C0C0C0",
  "Cd": "FFD98F",
  "In": "A67573",
  "Sn": "668080",
  "Sb": "9E63B5",
  "Te": "D47A00",
  "I": "940094",
  "Xe": "429EB0",
  "Cs": "57178F",
  "Ba": "00C900",
  "La": "70D4FF",
  "Ce": "FFFFC7",
  "Pr": "D9FFC7",
  "Nd": "C7FFC7",
  "Pm": "A3FFC7",
  "Sm": "8FFFC7",
  "Eu": "61FFC7",
  "Gd": "45FFC7",
  "Tb": "30FFC7",
  "Dy": "1FFFC7",
  "Ho": "00FF9C",
  "Er": "00E675",
  "Tm": "00D452",
  "Yb": "00BF38",
  "Lu": "00AB24",
  "Hf": "4DC2FF",
  "Ta": "4DA6FF",
  "W": "2194D6",
  "Re": "267DAB",
  "Os": "266696",
  "Ir": "175487",
  "Pt": "D0D0E0",
  "Au": "FFD123",
  "Hg": "B8B8D0",
  "Tl": "A6544D",
  "Pb": "575961",
  "Bi": "9E4FB5",
  "Po": "AB5C00",
  "At": "754F45",
  "Rn": "428296",
  "Fr": "420066",
  "Ra": "007D00",
  "Ac": "70ABFA",
  "Th": "00BAFF",
  "Pa": "00A1FF",
  "U": "008FFF",
  "Np": "0080FF",
  "Pu": "006BFF",
  "Am": "545CF2",
  "Cm": "785CE3",
  "Bk": "8A4FE3",
  "Cf": "A136D4",
  "Es": "B31FD4",
  "Fm": "B31FBA",
  "Md": "B30DA6",
  "No": "BD0D87",
  "Lr": "C70066",
  "Rf": "CC0059",
  "Db": "D1004F",
  "Sg": "D90045",
  "Bh": "E00038",
  "Hs": "E6002E",
  "Mt": "EB0026"
}

cmap = {k: '#' + v for k, v in cmap.items()}

from pymatgen.analysis.local_env import CrystalNN, BrunnerNN_reciprocal, MinimumDistanceNN
import networkx as nx




def compute_fig(struct: Structure, radius='uniform', show_bonds=True, rez=3):
    df = struct.as_dataframe()
    df['symbol'] = [e.elements[0].symbol for e in df['Species']]
    df['color'] = [cmap[s] for s in df['symbol']]
    if radius == 'uniform':
        rad = 0.5
        atoms = Spheres(df[['x', 'y', 'z']].values, r=rad, c=df['color'], res=rez)
    elif radius == 'atomic':
        df['radius'] = [e.elements[0].atomic_radius for e in df['Species']]

    # Create and show visual
    if radius != 'uniform':
        atoms_list = []
        # can't plot different colors and radii, have to make multiple plots
        for color, vals in df.groupby('color').groups.items():
            atoms_list.append(Spheres(df.loc[vals, ['x', 'y', 'z']].values, r=df.loc[vals, 'radius'], c=color, res=rez))

        atoms = Group(atoms_list)


    objs = [atoms]
    if show_bonds:
        graph = MinimumDistanceNN().get_bonded_structure(struct)
        subgraph = graph.graph

        lines = {c: ([], []) for c in pd.unique(df['color'])}
        for b1, b2, data in subgraph.edges(data=True):
            if data['to_jimage'] != (0, 0, 0):
                # bond between adjacent cells, don't show
                continue

            subs = df.loc[[b1, b2]]

            first = subs.iloc[0]
            second = subs.iloc[1]
            middle = subs[['x', 'y', 'z']].mean(axis=0).values

            a1, a2 = lines[first['color']]
            a1.append(first[['x', 'y', 'z']].values)
            a2.append(middle)

            a1, a2 = lines[second['color']]
            a1.append(middle)
            a2.append(second[['x', 'y', 'z']].values)

        bond_list = []
        for c, (a1, a2) in lines.items():
            bond_list.append(Lines(a1, a2, c=c, lw=1))

        bonds = Group(bond_list)
        objs.append(bonds)

    return show(*objs, interactive=False, viewup=(0.8, 0.8, 0.8), screensize=(800, 800))

def save_fig(struct, out_name, style, **kwargs):
    if style == 'spacefilling':
        kwargs.update(dict(radius='atomic', show_bonds=False))
    elif style == 'ballstick':
        kwargs.update(dict(radius='uniform', show_bonds=True))

    fig = compute_fig(struct, **kwargs)
    fig.save(out_name)

def visualize_cifs(
    input: Path,
    output: Path,
    style: str,
    c,
    **kwargs
):
    if not output.exists():
        output.mkdir()

    files = list(input.glob('*.cif'))

    with tqdm(files) as bar:
        for cif in files:
            name = cif.stem
            bar.set_description_str(cif.name)
            output_name = output / f'{name}.png'
            s = Structure.from_file(cif)
            s.make_supercell(c)
            save_fig(s, output_name, style, **kwargs)
            bar.update()


def scale_parse(s):
    nums = [int(n) for n in s.split(',')]
    if len(nums) == 1:
        return nums[0]
    elif len(nums) == 3:
        return tuple(nums)
    else:
        raise ValueError(f'Cannot parse {nums} as supercell: either 3,3,3 or 3 is acceptable')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, help='folder to get CIFs from')
    parser.add_argument('-s', '--style', choices=['spacefilling', 'ballstick'], help='structure drawing type', default='spacefilling')
    parser.add_argument('-o', '--output', type=Path, help='folder to put drawings in', default='images')
    parser.add_argument('-c', '--cell', type=scale_parse, help='supercell: format either as 3 or 3,3,3', default='1')
    parser.add_argument('-r', '--rez', type=int, help='resolution (0-10): more takes longer but is better', default='10')
    args = parser.parse_args()
    visualize_cifs(args.input, args.output, args.style, args.cell, rez=args.rez)