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
import zipfile

import click
import matplotlib
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from tqdm import tqdm

import skimage.feature
import skimage.morphology
import skimage.segmentation
from imageio import imread
from roi import ROIEncoder, ROIPolygon

try:
    import morphsnakes as ms
    USE_MS = True
    if not hasattr(ms, 'MorphGAC'):
        logging.warning('Morphsnakes is not properly installed. Using scikit image instead.')
        USE_MS = False
except ImportError:
    USE_MS = False



# in case you are running on machine without display, e.g. server
if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

def detect_ridges(gray, sigma=1.0):
    H_elems = skimage.feature.hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = skimage.feature.hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges

def ridges_likelihood(l2, l1, alpha=1000, beta=0.00000003):
    Rb = np.abs(l2/l1)
    S = l1*l1 + l2*l2
    res = np.exp(-Rb/alpha)*(1-np.exp(-(S*S)/beta))
    res[l1>0]=0
    return res

def preprocess(img):
    ridges = []
    for i in range(3, 20):
        a, b = detect_ridges(img, sigma=i)
        ridges.append(ridges_likelihood(a,b))
    result = np.amax(np.array(ridges), 0)
    result = skimage.morphology.closing(result, skimage.morphology.disk(3))
    return result

def visual_callback(background, fig=None, plot_iter=False):
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()

    nb_plots = 2 if plot_iter else 1

    ax1 = fig.add_subplot(1, nb_plots, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    if plot_iter:
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(background, cmap=plt.cm.gray)
        plt.pause(0.00001)

    def callback_iter(levelset):
        if ax2.collections:
            del ax2.collections[0]
        ax2.contour(levelset, [0.5], colors='r')
        fig.canvas.draw()
        plt.pause(0.00001)

    def callback_done(polygon):
        ax1.plot(polygon[:,1], polygon[:,0])
        # fig.canvas.draw()
        plt.pause(0.00001)

    if plot_iter:
        return callback_done, callback_iter
    else:
        return callback_done, lambda x: None

def segment(img, center=[0,0], radius=20, visual=lambda x: None, iterations=200, smoothing=6, threshold=0.65):
    # Initialization of the level-set.
    init_ls = skimage.segmentation.circle_level_set(img.shape, center, radius)

    if USE_MS:
        morph = ms.MorphGAC(init_ls, img, smoothing, threshold, 1)
        prev_ls = init_ls
        for i in range(iterations):
            morph.step()
            visual(morph.levelset)
            if i % 100 == 0:
                if (prev_ls != morph.levelset).sum() < 50:
                    break
                prev_ls = np.copy(morph.levelset)
        if i == 1000:
            logging.debug('Did not converge.')
        else:
            logging.debug('Converged after %d iterations.', i)
        segmentation = morph.levelset
    else:
        segmentation = skimage.segmentation.morphological_geodesic_active_contour(img,
                iterations=iterations, init_level_set=init_ls,
                smoothing=smoothing, threshold=threshold, balloon=1,
                iter_callback=visual)

    return segmentation

def as_polygon(mat):
    mat = mat.copy() != 0
    mat = np.bitwise_xor(mat, sp.ndimage.binary_erosion(mat))

    xs, ys = np.nonzero(mat)
    minx = xs[0]
    maxx = xs[-1]

    xlist = range(minx - 1, maxx + 2)
    tmp = np.searchsorted(xs, xlist, side="left")
    starts = dict(zip(xlist, tmp))
    tmp = np.searchsorted(xs, xlist, side="right")
    ends = dict(zip(xlist, tmp))

    unused = np.ones(len(xs), dtype=np.bool)
    vertex_loop = [(xs[0], ys[0])]
    unused[0] = 0
    count = 0
    while True:
        count += 1
        x, y = vertex_loop[-1]
        for i in range(starts[x - 1], ends[x + 1]):
            row = ys[i]
            if unused[i] and (row == y or row == y + 1 or row == y - 1):
                vertex_loop.append((xs[i], row))
                unused[i] = 0
                break
        else:
            if abs(x - xs[0]) <= 1 and abs(y - ys[0]) <= 1:
                break
            else:
                vertex_loop.pop()
    return np.array(vertex_loop)

def create_gimg(img, alpha=100, sigma=4.48):
    # g(I)
    gimg = skimage.segmentation.inverse_gaussian_gradient(img, alpha=100, sigma=2.48)
    return gimg

def zipdir(path):
    zipf = zipfile.ZipFile(f'{path}.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(path):
        for file in files:
            zipf.write(os.path.join(root, file), file)
    zipf.close()

@click.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('cell_locations', type=click.Path(exists=True))
@click.option('--radius', default=20, help='Initial cell radius')
@click.option('--visual', default=False, help='Show visual progress')
@click.option('--dopreprocess', default=False, help='Use an additional preprocssing step to highlight the ridges')
@click.option('--iterations', default=200, help='Show visual progress')
@click.option('--smoothing', default=6, help='Number of times the smoothing operator is applied per iteration. Reasonable values are around 1-4. Larger values lead to smoother segmentations.')
@click.option('--threshold', default=0.65, help='Areas of the image with a value smaller than this threshold will be considered borders. The evolution of the contour will stop in this areas.')
def cli(image_path, cell_locations, radius, dopreprocess, visual, iterations, smoothing, threshold):
    """A program that segements muscle fibers in immunofluorescent muscle cross-sections."""

    # Load the image.
    img = imread(image_path)[..., 1] / 255.0

    if dopreprocess:
        # Create a preprocessed version of the original image that enhances and
        # highlights the borders
        rimg = preprocess(img)
    else:
        rimg = img
    gimg = create_gimg(rimg)

    # Read a csv file listing the centers of the cells that should be segmented
    with open(cell_locations) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        cells = [{
            'id': row[' '], 
            'pos': (float(row['Y']), float(row['X']))
            } for row in csv_reader]

    # Create output directory
    output_path = os.path.splitext(image_path)[0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        yes = {'yes','y', 'ye', ''}
        no = {'no','n'}

        print(f'The directory "{output_path}" will be overriden. Continue? [y/n]')
        choice = input().lower()
        if choice in yes:
            shutil.rmtree(output_path)
            os.makedirs(output_path)
        else:
           return False

    # Process each cell
    callback_cell, callback_iter = visual_callback(img, plot_iter=visual)
    for cell in tqdm(cells):
        segmentation = segment(gimg, cell['pos'], radius, callback_iter, iterations, smoothing, threshold)
        polygon = as_polygon(segmentation)
        callback_cell(polygon)

        roi_obj = ROIPolygon(polygon[:,1], polygon[:,0])
        with ROIEncoder(f'{output_path}/{cell["id"]}.roi', roi_obj, cell["id"]) as roi:
            roi.write()

    zipdir(output_path)

    logging.info("Done.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cli()
