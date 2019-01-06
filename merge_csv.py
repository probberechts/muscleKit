"""
muscleKit Copyright (C) 2019 Pieter Robberchts

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
from pathlib import Path

import click
import pandas as pd


@click.command()
@click.argument('root_path', type=click.Path(exists=True))
@click.argument('out_path', type=click.Path(exists=False))
@click.option('--template', default='*colors.csv', help='regex for CSV files')
def cli(root_path, template):
    """Merge a set of CSV files into a single XLSX file with tabs."""
    csv_files = list(Path(root_path).rglob(template))
    writer = pd.ExcelWriter(out_path)
    for f in csv_files:
        df = pd.read_csv(f)
        df.to_excel(writer, sheet_name=os.path.dirname(f).split('/')[-1])
    writer.save()
