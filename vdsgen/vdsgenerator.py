#!/bin/env dls-python
"""A CLI tool for generating virtual datasets from individual HDF5 files."""

import os
import re
import logging

from collections import namedtuple

import h5py as h5

logging.basicConfig(level=logging.INFO)

Source = namedtuple("Source", ["frames", "height", "width", "dtype"])
VDS = namedtuple("VDS", ["shape", "spacing"])


class VDSGenerator(object):

    """A class to generate Virtual Datasets from raw HDF5 files."""

    # Constants
    CREATE = "w"  # Will overwrite any existing file
    APPEND = "a"
    READ = "r"
    FULL_SLICE = slice(None)

    # Default Values
    stripe_spacing = 10  # Pixel spacing between stripes in a module
    module_spacing = 10  # Pixel spacing between modules
    source_node = "data"  # Data node in source HDF5 files
    target_node = "full_frame"  # Data node in VDS file
    mode = CREATE  # Write mode for vds file

    def __init__(self, path, prefix=None, files=None, output=None, source=None,
                 source_node=None, target_node=None,
                 stripe_spacing=None, module_spacing=None):
        """
        Args:
            path(str): Root folder to find raw files and create VDS
            prefix(str): Prefix of HDF5 files to generate from
                e.g. image_ for image_1.hdf5, image_2.hdf5, image_3.hdf5
            files(list(str)): List of HDF5 files to generate from
            output(str): Name of VDS file.
            source(dict): Height, width, data_type and frames for source data
                Provide this to create a VDS for raw files that don't exist yet
            source_node(str): Data node in source HDF5 files
            target_node(str): Data node in VDS file
            stripe_spacing(int): Spacing between stripes in module
            module_spacing(int): Spacing between modules

        """
        if (prefix is None and files is None) or \
                (prefix is not None and files is not None):
            raise ValueError("One, and only one, of prefix or files required.")

        self.path = path

        # Overwrite default values with arguments, if given
        if source_node is not None:
            self.source_node = source_node
        if target_node is not None:
            self.target_node = target_node
        if stripe_spacing is not None:
            self.stripe_spacing = stripe_spacing
        if module_spacing is not None:
            self.module_spacing = module_spacing

        # If Files not given, find files using path and prefix.
        if files is None:
            self.prefix = prefix
            self.datasets = self.find_files()
            files = [path_.split("/")[-1] for path_ in self.datasets]
        # Else, get common prefix of given files and store full path
        else:
            self.prefix = os.path.commonprefix(files)
            self.datasets = [os.path.join(path, file_) for file_ in files]

        # If output vds file name given, use, otherwise generate a default
        if output is None:
            self.name = self.construct_vds_name(files)
        else:
            self.name = output

        # If source not given, check files exist and get metadata.
        if source is None:
            for file_ in self.datasets:
                if not os.path.isfile(file_):
                    raise IOError(
                        "File {} does not exist. To create VDS from raw "
                        "files that haven't been created yet, source "
                        "must be provided.".format(file_))
            self.source_metadata = self.process_source_datasets()
        # Else, store given source metadata
        else:
            frames, height, width = self.parse_shape(source['shape'])
            self.source_metadata = Source(
                frames=frames, height=height, width=width,
                dtype=source['dtype'])

        self.output_file = os.path.abspath(os.path.join(self.path, self.name))

    @staticmethod
    def parse_shape(shape):
        """Split shape into height, width and frames.

        Args:
            shape(tuple): Shape of dataset

        Returns:
            frames, height, width

        """
        # The last two elements of shape are the height and width of the image
        height, width = shape[-2:]
        # Everything before that is the frames for each axis
        frames = shape[:-2]

        return frames, height, width

    def generate_vds(self):
        """Generate a virtual dataset."""
        if os.path.isfile(self.output_file):
            with h5.File(self.output_file, self.READ, libver="latest") as vds:
                node = vds.get(self.target_node)
            if node is not None:
                raise IOError("VDS {file} already has an entry for node "
                              "{node}".format(file=self.output_file,
                                              node=self.target_node))
            else:
                self.mode = self.APPEND

        file_names = [file_.split('/')[-1] for file_ in self.datasets]
        logging.info("Combining datasets %s into %s",
                     ", ".join(file_names), self.name)

        vds_data = self.construct_vds_metadata(self.source_metadata)
        map_list = self.create_vds_maps(self.source_metadata, vds_data)

        logging.info("Creating VDS at %s", self.output_file)
        with h5.File(self.output_file, self.mode, libver="latest") as vds:
            self.validate_node(vds)
            vds.create_virtual_dataset(VMlist=map_list, fillvalue=0x1)

        logging.info("Creation successful!")

    def find_files(self):
        """Find HDF5 files in given folder with given prefix.

        Returns:
            list: HDF5 files in folder that have the given prefix

        """
        regex = re.compile(self.prefix + r"\d+\.(hdf5|hdf|h5)")

        files = []
        for file_ in sorted(os.listdir(self.path)):
            if re.match(regex, file_):
                files.append(os.path.abspath(os.path.join(self.path, file_)))

        if len(files) == 0:
            raise IOError("No files matching pattern found. Got path: {path}, "
                          "prefix: {prefix}".format(path=self.path,
                                                    prefix=self.prefix))
        elif len(files) < 2:
            raise IOError("Folder must contain more than one matching HDF5 "
                          "file.")
        else:
            return files

    def construct_vds_name(self, files):
        """Generate the file name for the VDS from the sub files.

        Args:
            files(list(str)): HDF5 files being combined

        Returns:
            str: Name of VDS file

        """
        _, ext = os.path.splitext(files[0])
        vds_name = "{prefix}vds{ext}".format(prefix=self.prefix, ext=ext)

        return vds_name

    def grab_metadata(self, file_path):
        """Grab data from given HDF5 file.

        Args:
            file_path(str): Path to HDF5 file

        Returns:
            dict: Number of frames, height, width and data type of datasets

        """
        h5_data = h5.File(file_path, 'r')[self.source_node]
        frames, height, width = self.parse_shape(h5_data.shape)
        data_type = h5_data.dtype

        return dict(frames=frames, height=height, width=width, dtype=data_type)

    def process_source_datasets(self):
        """Grab data from the given HDF5 files and check for consistency.

        Returns:
            Source: Number of datasets and the attributes of them (frames,
                height width and data type)

        """
        data = self.grab_metadata(self.datasets[0])
        for dataset in self.datasets[1:]:
            temp_data = self.grab_metadata(dataset)
            for attribute, value in data.items():
                if temp_data[attribute] != value:
                    raise ValueError("Files have mismatched "
                                     "{}".format(attribute))

        return Source(frames=data['frames'], height=data['height'],
                      width=data['width'], dtype=data['dtype'])

    def construct_vds_metadata(self, source):
        """Construct VDS data attributes from source attributes.

        Args:
            source(Source): Attributes of data sets

        Returns:
            VDS: Shape, dataset spacing and output path of virtual data set

        """
        stripes = len(self.datasets)
        spacing = [0] * stripes
        for idx in range(0, stripes - 1, 2):
            spacing[idx] = self.stripe_spacing
        for idx in range(1, stripes, 2):
            spacing[idx] = self.module_spacing
        # We don't want the final stripe to have a gap afterwards
        spacing[-1] = 0

        height = (source.height * stripes) + sum(spacing)
        shape = source.frames + (height, source.width)

        return VDS(shape=shape, spacing=spacing)

    def create_vds_maps(self, source, vds_data):
        """Create a list of VirtualMaps of raw data to the VDS.

        Args:
            source(Source): Source attributes
            vds_data(VDS): VDS attributes

        Returns:
            list(VirtualMap): Maps describing links between raw data and VDS

        """
        source_shape = source.frames + (source.height, source.width)
        vds = h5.VirtualTarget(self.output_file, self.target_node,
                               shape=vds_data.shape)

        map_list = []
        current_position = 0
        for idx, dataset in enumerate(self.datasets):
            logging.info("Processing dataset %s", idx + 1)

            v_source = h5.VirtualSource(dataset, self.source_node,
                                        shape=source_shape)

            start = current_position
            stop = start + source.height + vds_data.spacing[idx]
            current_position = stop

            index = tuple([self.FULL_SLICE] * len(source.frames) +
                          [slice(start, stop)] + [self.FULL_SLICE])
            v_target = vds[index]
            v_map = h5.VirtualMap(v_source, v_target, dtype=source.dtype)
            map_list.append(v_map)

        return map_list

    def validate_node(self, vds_file):
        """Check if it is possible to create the given node.

        Check the target node is valid (no leading or trailing slashes)
        Create any sub-group of the target node if it doesn't exist.

        Args:
            vds_file(h5py.File): File to check for node

        """
        if self.target_node.startswith("/") or self.target_node.endswith("/"):
            raise ValueError("Target node should have no leading or trailing "
                             "slashes, got {}".format(self.target_node))

        if "/" in self.target_node:
            sub_group = self.target_node.rsplit("/", 1)[0]
            if vds_file.get(sub_group) is None:
                vds_file.create_group(sub_group)
