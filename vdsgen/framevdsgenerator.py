#!/bin/env dls-python
"""A class for generating virtual dataset frames from sub-frames."""

import logging

import h5py as h5

from vdsgenerator import VDSGenerator, SourceMeta


class FrameVDSGenerator(VDSGenerator):

    """A class to generate Virtual Dataset frames from sub-frames."""

    logger = logging.getLogger("FrameVDSGenerator")

    def __init__(self, path, prefix=None, files=None, output=None, source=None,
                 source_node=None, target_node=None, fill_value=None,
                 log_level=None):
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
            fill_value(int): Fill value for spacing
            log_level(int): Logging level (off=3, info=2, debug=1) -
                Default is info

        """
        self.total_frames = 0

        super(FrameVDSGenerator, self).__init__(
            path, prefix, files, output, source, source_node, target_node,
            fill_value, log_level)

    def process_source_datasets(self):
        """Grab data from the given HDF5 files and check for consistency.

        Returns:
            Source: Number of datasets and the attributes of them (frames,
                height width and data type)

        """
        data = self.grab_metadata(self.datasets[0])
        self.total_frames = data["frames"][0]
        for dataset in self.datasets[1:]:
            temp_data = self.grab_metadata(dataset)
            self.total_frames += temp_data["frames"][0]
            for attribute, value in data.items():
                if attribute != "frames" and temp_data[attribute] != value:
                    raise ValueError("Files have mismatched "
                                     "{}".format(attribute))

        source = SourceMeta(frames=data['frames'],
                            height=data['height'], width=data['width'],
                            dtype=data['dtype'])

        self.logger.debug("Source metadata retrieved:\n"
                          "  Frames: %s\n"
                          "  Height: %s\n"
                          "  Width: %s\n"
                          "  Data Type: %s", self.total_frames, *source[1:])
        return source

    def create_vds_maps(self, source_meta):
        """Create a list of VirtualMaps of raw data to the VDS.

        Args:
            source_meta(SourceMeta): Source attributes

        Returns:
            list(VirtualMap): Maps describing links between raw data and VDS

        """
        source_dims = (source_meta.height, source_meta.width)
        target_shape = (self.total_frames,
                        source_meta.height, source_meta.width)
        self.logger.debug("VDS metadata:\n"
                          "  Shape: %s\n", target_shape)

        vds = h5.VirtualTarget(self.output_file, self.target_node,
                               shape=target_shape)

        total_datasets = len(self.datasets)

        map_list = []
        for dataset_idx, dataset in enumerate(self.datasets):
            dataset_frames = h5.File(dataset)[self.source_node].shape[0]
            source = h5.VirtualSource(dataset, self.source_node,
                                      shape=(dataset_frames,)+source_dims)

            for frame_idx in range(dataset_frames):

                # Hyperslab: Frame[frame_idx],
                #            Full slice for height and width
                source_hyperslab = tuple([slice(frame_idx, frame_idx + 1)] +
                                         [self.FULL_SLICE, self.FULL_SLICE])
                v_source = source[source_hyperslab]

                # Hyperslab: Single frame based on indexes,
                #            Full slice for height and width
                target_idx = dataset_idx + total_datasets * frame_idx
                vds_hyperslab = tuple([slice(target_idx, target_idx + 1)] +
                                      [self.FULL_SLICE, self.FULL_SLICE])
                v_target = vds[vds_hyperslab]

                v_map = h5.VirtualMap(v_source, v_target,
                                      dtype=source_meta.dtype)

                self.logger.debug("Mapping frame %s of %s to %s of %s.",
                                  frame_idx, dataset.split("/")[-1],
                                  target_idx, self.name)
                map_list.append(v_map)

        return map_list
