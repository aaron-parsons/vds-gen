#!/bin/env dls-python
"""A CLI tool for generating virtual datasets from individual HDF5 files."""

import os
import sys
from argparse import ArgumentParser
import re
import logging

from collections import namedtuple

import h5py as h5

logging.basicConfig(level=logging.INFO)

Source = namedtuple("Source",
                    ["datasets", "frames", "height", "width", "dtype"])
VDS = namedtuple("VDS", ["shape", "spacing", "path"])

INTER_MODULE_SPACING = 10  # Pixel spacing between stripes in a module
MODULE_SPACING = 10  # Pixel spacing between modules


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("path", type=str,
                        help="Path to folder containing HDF5 files.")
    parser.add_argument("prefix", type=str,
                        help="Root name of images - e.g 'stripe_' to combine "
                             "the images 'stripe_1.hdf5', 'stripe_2.hdf5' "
                             "and 'stripe_3.hdf5' located at <path>.")

    return parser.parse_args()


def find_files(path, prefix):
    """Find HDF5 files in given folder with given prefix.

    Args:
        path(str): Path to folder containing files
        prefix(str): Root name of image files

    Returns:
        list: HDF5 files in folder that have the given prefix

    """
    regex = re.compile(prefix + r"\d\.(hdf5|h5)")

    files = []
    for file_ in sorted(os.listdir(path)):
        if re.match(regex, file_):
            files.append(os.path.join(path, file_))

    if len(files) == 0:
        raise IOError("No files matching pattern found.")
    elif len(files) < 2:
        raise IOError("Folder must contain more than one matching HDF5 file.")
    else:
        return files


def construct_vds_name(prefix, files):
    """Generate the file name for the VDS from the sub files.

    Args:
        prefix(str): Root name of image files
        files(list(str)): HDF5 files being combined

    Returns:
        str: Name of VDS file

    """
    _, ext = os.path.splitext(files[0])
    vds_name = "{prefix}vds{ext}".format(prefix=prefix, ext=ext)

    return vds_name


def grab_metadata(file_path):
    """Grab data from given HDF5 file.

    Args:
        file_path(str): Path to HDF5 file

    Returns:
        dict: Number of frames, height, width and data type of datasets

    """
    h5_data = h5.File(file_path, 'r')["data"]
    frames, height, width = h5_data.shape
    data_type = h5_data.dtype

    return dict(frames=frames, height=height, width=width, dtype=data_type)


def process_source_datasets(datasets):
    """Grab data from the given HDF5 files and check for consistency.

    Args:
        datasets(list(str)): Datasets to grab data from

    Returns:
        Source: Number of datasets and the attributes of them (frames, height
            width and data type)

    """
    data = grab_metadata(datasets[0])
    for path in datasets[1:]:
        temp_data = grab_metadata(path)
        for attribute, value in data.items():
            if temp_data[attribute] != value:
                raise ValueError("Files have mismatched {}".format(attribute))

    return Source(frames=data['frames'], height=data['height'],
                  width=data['width'], dtype=data['dtype'], datasets=datasets)


def construct_vds_metadata(source, output_file):
    """Construct VDS data attributes from source attributes.

    Args:
        source(Source): Attributes of data sets
        output_file(str): File path of new VDS

    Returns:
        VDS: Shape, dataset spacing and output path of virtual data set

    """
    stripes = len(source.datasets)

    spacing = [0] * stripes
    for idx in range(0, stripes - 1, 2):
        spacing[idx] = INTER_MODULE_SPACING
    for idx in range(1, stripes, 2):
        spacing[idx] = MODULE_SPACING
    spacing[-1] = 0  # We don't want the final stripe to have a gap afterwards

    height = (source.height * stripes) + sum(spacing)
    shape = (source.frames, height, source.width)

    return VDS(shape=shape, spacing=spacing, path=output_file)


def create_vds_maps(source, vds_data):
    """Create a list of VirtualMaps of raw data to the VDS.

    Args:
        source(Source): Source attributes
        vds_data(VDS): VDS attributes

    Returns:
        list(VirtualMap): Maps describing links between raw data and VDS

    """
    source_shape = (source.frames, source.height, source.width)
    vds = h5.VirtualTarget(vds_data.path, "full_frame", shape=vds_data.shape)

    map_list = []
    current_position = 0
    for idx, dataset in enumerate(source.datasets):
        logging.info("Processing dataset %s", idx + 1)

        v_source = h5.VirtualSource(dataset, "data", shape=source_shape)

        start = current_position
        stop = start + source.height + vds_data.spacing[idx]
        current_position = stop

        v_target = vds[:, start:stop, :]
        v_map = h5.VirtualMap(v_source, v_target, dtype=source.dtype)
        map_list.append(v_map)

    return map_list


def generate_vds(path, prefix):
    """Generate a virtual dataset.

    Args:
        path(str): Path to folder containing HDF5 files
        prefix(str): Prefix of HDF5 files to generate from (in <path> folder)
            e.g. image_ for image_1.hdf5, image_2.hdf5, image_3.hdf5

    """
    file_paths = find_files(path, prefix)
    vds_name = construct_vds_name(prefix, file_paths)
    output_file = os.path.join(path, vds_name)

    file_names = [file_.split('/')[-1] for file_ in file_paths]
    logging.info("Combining datasets %s into %s",
                 ", ".join(file_names), vds_name)

    source = process_source_datasets(file_paths)
    vds_data = construct_vds_metadata(source, output_file)
    map_list = create_vds_maps(source, vds_data)

    logging.info("Creating VDS at %s", output_file)
    with h5.File(output_file, "w", libver="latest") as vds_file:
        vds_file.create_virtual_dataset(VMlist=map_list, fill_value=0x1)

    logging.info("Creation successful!")


def main():
    """Run program."""
    args = parse_args()
    generate_vds(args.path, args.prefix)

if __name__ == "__main__":
    sys.exit(main())
