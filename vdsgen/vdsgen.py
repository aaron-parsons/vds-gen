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

STRIPE_SPACING = 10  # Pixel spacing between stripes in a module
MODULE_SPACING = 10  # Pixel spacing between modules
DATA_PATH = "data"   # Location of data in HDF5 file tree


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument("path", type=str,
                        help="Path to folder containing HDF5 files.")

    # Definition of file names in <path> - Common prefix or explicit list
    file_definition = parser.add_mutually_exclusive_group(required=True)
    file_definition.add_argument(
        "-p", "--prefix", type=str, default=None, dest="prefix",
        help="Prefix of files - e.g 'stripe_' to combine the images "
             "'stripe_1.hdf5', 'stripe_2.hdf5' and 'stripe_3.hdf5' located "
             "at <path>.")
    file_definition.add_argument(
        "-f", "--files", nargs="*", type=str, default=None, dest="files",
        help="Manually define files to combine.")

    # Arguments required to allow VDS to be created before raw files exist
    parser.add_argument(
        "-e", "--empty", action="store_true", dest="empty",
        help="Make empty VDS pointing to datasets that don't exist, yet.")
    source_metadata = parser.add_argument_group()
    source_metadata.add_argument(
        "--frames", type=int, default=1, dest="frames",
        help="Number of frames to combine into VDS.")
    source_metadata.add_argument(
        "--height", type=int, default=256, dest="height",
        help="Height of raw datasets.")
    source_metadata.add_argument(
        "--width", type=int, default=1024, dest="width",
        help="Width of raw datasets.")
    source_metadata.add_argument(
        "--data_type", type=str, default="uint16", dest="data_type",
        help="Data type of raw datasets.")

    # Arguments to override defaults
    parser.add_argument("-s", "--stripe_spacing", nargs="?", type=int,
                        default=None, dest="stripe_spacing",
                        help="Spacing between two stripes in a module.")
    parser.add_argument("-m", "--module_spacing", nargs="?", type=int,
                        default=None, dest="module_spacing",
                        help="Spacing between two modules.")
    parser.add_argument("-d", "--data_path", nargs="?", type=str, default=None,
                        dest="data_path",
                        help="Data location in HDF5 files.")

    args = parser.parse_args()

    if args.empty and args.files is None:
        parser.error(
            "To make an empty VDS you must explicitly define --files for the "
            "eventual raw datasets.")
    if args.files is not None and len(args.files) < 2:
        parser.error("Must define at least two files to combine.")

    return args


def find_files(path, prefix):
    """Find HDF5 files in given folder with given prefix.

    Args:
        path(str): Path to folder containing files
        prefix(str): Root name of image files

    Returns:
        list: HDF5 files in folder that have the given prefix

    """
    regex = re.compile(prefix + r"\d+\.(hdf5|hdf|h5)")

    files = []
    for file_ in sorted(os.listdir(path)):
        if re.match(regex, file_):
            files.append(os.path.abspath(os.path.join(path, file_)))

    if len(files) == 0:
        raise IOError("No files matching pattern found. Got path: {path}, "
                      "prefix: {prefix}".format(path=path, prefix=prefix))
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


def grab_metadata(file_path, data_path):
    """Grab data from given HDF5 file.

    Args:
        file_path(str): Path to HDF5 file
        data_path(str): Location of raw data in HDF5 file

    Returns:
        dict: Number of frames, height, width and data type of datasets

    """
    h5_data = h5.File(file_path, 'r')[data_path]
    frames, height, width = h5_data.shape
    data_type = h5_data.dtype

    return dict(frames=frames, height=height, width=width, dtype=data_type)


def process_source_datasets(datasets, data_path):
    """Grab data from the given HDF5 files and check for consistency.

    Args:
        datasets(list(str)): Datasets to grab data from
        data_path(str): Location of raw data in HDF5 file

    Returns:
        Source: Number of datasets and the attributes of them (frames, height
            width and data type)

    """
    data = grab_metadata(datasets[0], data_path)
    for path in datasets[1:]:
        temp_data = grab_metadata(path, data_path)
        for attribute, value in data.items():
            if temp_data[attribute] != value:
                raise ValueError("Files have mismatched {}".format(attribute))

    return Source(frames=data['frames'], height=data['height'],
                  width=data['width'], dtype=data['dtype'], datasets=datasets)


def construct_vds_metadata(source, output_file,
                           stripe_spacing=None, module_spacing=None):
    """Construct VDS data attributes from source attributes.

    Args:
        source(Source): Attributes of data sets
        output_file(str): File path of new VDS
        stripe_spacing(int): Spacing between stripes in module
        module_spacing(int): Spacing between modules

    Returns:
        VDS: Shape, dataset spacing and output path of virtual data set

    """
    if stripe_spacing is None:
        stripe_spacing = STRIPE_SPACING
    if module_spacing is None:
        module_spacing = MODULE_SPACING

    stripes = len(source.datasets)
    spacing = [0] * stripes
    for idx in range(0, stripes - 1, 2):
        spacing[idx] = stripe_spacing
    for idx in range(1, stripes, 2):
        spacing[idx] = module_spacing
    spacing[-1] = 0  # We don't want the final stripe to have a gap afterwards

    height = (source.height * stripes) + sum(spacing)
    shape = (source.frames, height, source.width)

    return VDS(shape=shape, spacing=spacing, path=output_file)


def create_vds_maps(source, vds_data, data_path):
    """Create a list of VirtualMaps of raw data to the VDS.

    Args:
        source(Source): Source attributes
        vds_data(VDS): VDS attributes
        data_path(str): Path to raw data in HDF5 file

    Returns:
        list(VirtualMap): Maps describing links between raw data and VDS

    """
    source_shape = (source.frames, source.height, source.width)
    vds = h5.VirtualTarget(vds_data.path, "full_frame", shape=vds_data.shape)

    map_list = []
    current_position = 0
    for idx, dataset in enumerate(source.datasets):
        logging.info("Processing dataset %s", idx + 1)

        v_source = h5.VirtualSource(dataset, data_path, shape=source_shape)

        start = current_position
        stop = start + source.height + vds_data.spacing[idx]
        current_position = stop

        v_target = vds[:, start:stop, :]
        v_map = h5.VirtualMap(v_source, v_target, dtype=source.dtype)
        map_list.append(v_map)

    return map_list


def generate_vds(path, prefix=None, files=None, source=None, data_path=None,
                 stripe_spacing=None, module_spacing=None):
    """Generate a virtual dataset.

    Args:
        path(str): Path to folder containing HDF5 files
        prefix(str): Prefix of HDF5 files to generate from (in <path> folder)
            e.g. image_ for image_1.hdf5, image_2.hdf5, image_3.hdf5
        files(list(str)): List of files to combine.
        source(dict): Height, width, data_type and frames for source data
        data_path(str): Path to raw data in HDF5 file
        stripe_spacing(int): Spacing between stripes in module
        module_spacing(int): Spacing between modules

    """
    if (prefix is None and files is None) or \
            (prefix is not None and files is not None):
        raise ValueError("One, and only one, of prefix or files required.")

    if data_path is None:
        data_path = DATA_PATH

    if files is None:
        file_paths = find_files(path, prefix)
        files = [path_.split("/")[-1] for path_ in file_paths]
    else:
        file_paths = [os.path.join(path, file_) for file_ in files]
        prefix = os.path.commonprefix(files)

    vds_name = construct_vds_name(prefix, files)
    output_file = os.path.abspath(os.path.join(path, vds_name))

    file_names = [file_.split('/')[-1] for file_ in file_paths]
    logging.info("Combining datasets %s into %s",
                 ", ".join(file_names), vds_name)

    if source is None:
        source_metadata = process_source_datasets(file_paths, data_path)
    else:
        source_metadata = Source(
            frames=source['frames'], height=source['height'],
            width=source['width'], dtype=source['dtype'], datasets=file_paths)

    vds_data = construct_vds_metadata(source_metadata, output_file,
                                      stripe_spacing, module_spacing)
    map_list = create_vds_maps(source_metadata, vds_data, data_path)

    logging.info("Creating VDS at %s", output_file)
    with h5.File(output_file, "w", libver="latest") as vds_file:
        vds_file.create_virtual_dataset(VMlist=map_list, fill_value=0x1)

    logging.info("Creation successful!")


def main():
    """Run program."""
    args = parse_args()

    if args.empty:
        source_metadata = dict(frames=args.frames, height=args.height,
                               width=args.width, dtype=args.data_type)
    else:
        source_metadata = None

    generate_vds(args.path, args.prefix, args.files, source_metadata,
                 args.data_path, args.stripe_spacing, args.module_spacing)

if __name__ == "__main__":
    sys.exit(main())
