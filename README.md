# muscleKit

A set of Python scripts to automate the analysis of immunofluorescent muscle cross-sections. Currently, the following tasks are supported:

 - Semi-automatic fiber segmentation and cross-sectional analysis
 - Fiber typing

 These scripts integrate with the [ImageJ](https://imagej.nih.gov) image processing software for easy pre and post processing.

![muscleKit](https://www.dropbox.com/s/65q4vzkaqbmrf1v/muscleKit.png?raw=1)

## Installation

### Requirements

- Python 3.6
- [ImageJ](https://imagej.nih.gov/ij/download.html) (optional)

### Setup

Clone the repository and install the required dependencies.

```shell
git clone https://github.com/probberechts/muscleKit.git
cd muscleKit
pip install -r requirements.txt
```

By default, the fiber segmentation script will use [the Morphological Snakes implementation of scikit-image](http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_morphsnakes.html). However, a muck faster C++ based implementation is available in [the development branch of the morphsnakes package](https://github.com/pmneila/morphsnakes/tree/develop). To use this implementation instead, you should run the commands below:

```shell
cd muscleKit
git clone -b develop https://github.com/pmneila/morphsnakes.git
cd morphsnakes
pip install Cython
pip install .
```

## Usage

### Fiber Segmentation

[![Screencast](https://www.dropbox.com/s/txfteyinjoumkw0/fiber_segmentation_screen.png?raw=1)](https://www.dropbox.com/s/ti5qs1vi9s8827z/fiber_segmentation_HQ.mov?raw=1)

1. Open the image which you would like to analyze with ImageJ.
2. Select the multi-point selection tool and click in the middle of each cell which you would like to segment.
3. Use the `export_multipoinset` macro to export these points as a CSV file.
4. Run the `fiber_segmentation.py` script to segment the fibers.

    ```shell
    python fiber_segmentation.py examples/test.tiff examples/test.tiff.csv --threshold 0.58
    ```

    Run `python fiber_segmentation.py -h` for an explanation with all parameters.
5. Once the script finishes, a ZIP file `test.zip` is created. This archive contains all ROI's of the selected cells and can be opened in ImageJ.

### Fiber Typing

[![Screencast](https://www.dropbox.com/s/z9wrqeqoayyn0ii/fiber_typing_screen.png?raw=1)](https://www.dropbox.com/s/s5iye8zrn2r6uix/fiber_typing_HQ.mov?raw=1)

1. Perform the fiber segmentation step and obtain all ROIs.
2. Open the image which you would like to analyze with ImageJ.
3. Use the ROI manager  (`Analyze > Tools > ROI Manager`) to open the corresponding ROIs (`ROI Manager> More > Open ...`).
4. Make sure ImageJ uses the file names as labels (`ROI Manager > More > Labels`).
5. Use ImageJ's measure feature (`ROI Manager > measure`) to create a CSV file with the label of each ROI. Optionally, you can use this tool to measure other features as well, such as the cross-sectional area. 
6. Adjust the color balance of the image (`Image > Adjust > Color Balance`) until each fiber type has a maximally different color.
7. Save the resulting image.
8. Run the fiber typing script.

    ```shell
    python fiber_typing.py examples/test.colors.tiff examples/test examples/test.analysis.txt
    ```

9. Adjust the sliders until you obtain a feasible configuration and click 'Done'.
10. The script will create a new CSV file, containing all fields in `test.analysis.txt` with an additional column containing the color of each fiber.

## Auhors

- Pieter Robberechts,  <https://people.cs.kuleuven.be/pieter.robberechts>
- Ruben Robberechts,  <rubenrobberechts@outlook.com>

## License

**[GNU GPL license](https://opensource.org/licenses/GPL-3.0)**
