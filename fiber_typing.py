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

import csv
import logging
import os
import shutil
import time

import click
import matplotlib
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import Button, RadioButtons, Slider
from tqdm import tqdm

from imageio import imread
from roi import ROIDecoder, ROIPolygon
from skimage.color import rgba2rgb


def visual_callback(background, cells, out=None, fig=None):
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()

    fig.clf()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.imshow(background, cmap=plt.cm.gray)
    plt.axis('off')
    plt.subplots_adjust(bottom=0.35)

    annotations = {}
    for cell in cells:
        annotations[cell['id']] = ax1.annotate(cell['color'], cell['pos'], color='white')

    axcolor = 'lightgoldenrodyellow'
    axblack = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
    axgreen = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
    axblue = plt.axes([0.2, 0.2, 0.65, 0.03], facecolor=axcolor)

    delta_f = 0.05
    sblue = Slider(axblue, 'Blue', 0.0, 1.0, valinit=0.4, valstep=delta_f)
    sgreen = Slider(axgreen, 'Green', 0.0, 1.0, valinit=0.2, valstep=delta_f)
    sblack = Slider(axblack, 'Black', 0.0, 1.0, valinit=0.25, valstep=delta_f)

    def update(val):
        black = sblack.val
        green = sgreen.val
        blue = sblue.val
        for cell in cells:
            r, g, b = cell['avg_colors']
            if g > green:
                cell['color'] = 'G'
            elif b > blue:
                cell['color'] = 'B'
            elif r + g + b < black:
                cell['color'] = 'Z'
            else:
                cell['color'] = '?'
            annotations[cell['id']].set_text(cell['color'])
        # l.set_ydata(amp*np.sin(2*np.pi*freq*t))
        # fig.canvas.draw_idle()
    update(None)
    sblack.on_changed(update)
    sgreen.on_changed(update)
    sblue.on_changed(update)

    doneax = plt.axes([0.8, 0.035, 0.1, 0.04])
    button = Button(doneax, 'Done', color=axcolor, hovercolor='0.975')

    def done(event):
        plt.close()
        if out is not None:
            fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
            ax.imshow(background, cmap=plt.cm.gray)
            plt.axis('off')
            for cell in cells:
                ax.annotate(cell['color'], cell['pos'], color='white', fontsize=2)
            fig.savefig(out, dpi=900)
            plt.close(fig)
    button.on_clicked(done)

    plt.show()
    return cells

def get_average_color(image, mask):
    """ Returns a 3-tuple containing the RGB value of the average color of the
    given square bounded area of length = n whose origin (top left corner) 
    is (x, y) in the given image"""
 
    r, g, b = 0, 0, 0
    count = 0

    ny, nx = mask.shape
    for i in range(ny):
        for j in range(nx):
            if mask[i, j]:
                pixlr, pixlg, pixlb = image[i, j]
                r += pixlr
                g += pixlg
                b += pixlb
                count += 1
    return ((r/count), (g/count), (b/count))

def polygon2mask(poly_verts, img):
    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    min_x, max_x = np.amin(poly_verts[:,0]), np.amax(poly_verts[:,0])
    nx = max_x - min_x
    min_y, max_y = np.amin(poly_verts[:,1]), np.amax(poly_verts[:,1])
    ny = max_y - min_y
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T

    poly_verts = poly_verts - np.array([min_x, min_y])
    path = Path(poly_verts)
    grid = path.contains_points(points)
    grid = grid.reshape((ny,nx))

    img_masked = np.copy(img[min_y:max_y,min_x:max_x])
    img_masked[~grid] = 0

    return img_masked, grid

@click.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('roi_path', type=click.Path(exists=True))
@click.argument('cell_data', type=click.Path(exists=True))
def cli(image_path, roi_path, cell_data):
    """A program that performs muscle fiber typing based on color analysis of immunofluorescent muscle cross-sections."""

    # Load the image.
    img = imread(image_path)
    ny, nx, c = img.shape
    if c == 4:
        img_rgb = rgba2rgb(img)
    else:
        img_rgb = img / 255
    
    # roi_path = os.path.splitext(image_path)[0]

    # Read a csv file listing the centers of the cells that should be segmented
    with open(cell_data) as csv_file:
        cells = [{k: v for k, v in row.items()}
                for row in csv.DictReader(csv_file, skipinitialspace=True)]
        csv_reader = csv.DictReader(csv_file, delimiter=',')

   # Process each cell
    for cell in tqdm(cells):
        cell['id'] = cell['Label'].split(':')[1]
        # Load the ROIs
        with ROIDecoder(f'{roi_path}/{cell["id"]}.roi') as roi:
            roi_obj = roi.get_roi()

        ny, nx, _ = img.shape
        poly_verts = np.array(roi_obj.points)
        img_masked, mask = polygon2mask(poly_verts, img_rgb)

        cell['pos'] = np.array(roi_obj.points).mean(axis=0)
        cell['avg_colors'] = get_average_color(img_masked, mask)
        cell['color'] = '?'

    # Tune color thresholds
    result = visual_callback(img_rgb, cells, f'{os.path.splitext(cell_data)[0]}_colors.png')

    # Write result
    keys = list(result[0].keys())
    cell_data_out = f'{os.path.splitext(cell_data)[0]}_colors.csv'
    with open(cell_data_out, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(result)

    logging.info("Done.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cli()
