import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from vdsgenerator import VDSGenerator
from framevdsgenerator import FrameVDSGenerator
from subframevdsgenerator import SubFrameVDSGenerator

help_message = """
-------------------------------------------------------------------------------
A script to create a virtual dataset composed of multiple raw HDF5 files.

The minimum required arguments are <path> and either -p <prefix> or -f <files>.

For example:

 > ../vdsgen/app.py /scratch/images -p stripe_
 > ../vdsgen/app.py /scratch/images -f stripe_1.hdf5 stripe_2.hdf5

You can create an empty VDS, for raw files that don't exist yet, with the -e
flag; you will then need to provide --shape and --data_type, though defaults
are provided for these.
-------------------------------------------------------------------------------
"""


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(usage=help_message,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "path", type=str, help="Root folder of source files and VDS.")

    # Definition of file names in <path> - Common prefix or explicit list
    file_definition = parser.add_mutually_exclusive_group(required=True)
    file_definition.add_argument(
        "-p", "--prefix", type=str, default=None, dest="prefix",
        help="Prefix of files to search <path> for - e.g 'stripe_' to combine "
             "'stripe_1.hdf5' and 'stripe_2.hdf5'.")
    file_definition.add_argument(
        "-f", "--files", nargs="*", type=str, default=None, dest="files",
        help="Explicit names of raw files in <path>.")

    # Arguments required to allow VDS to be created before raw files exist
    empty_vds = parser.add_argument_group()
    empty_vds.add_argument(
        "-e", "--empty", action="store_true", dest="empty",
        help="Make empty VDS pointing to datasets that don't exist yet.")
    empty_vds.add_argument(
        "--shape", type=int, nargs="*", default=[1, 256, 2048], dest="shape",
        help="Shape of dataset - 'frames height width', where frames is N "
             "dimensional.")
    empty_vds.add_argument(
        "-t", "--data_type", type=str, default="uint16", dest="data_type",
        help="Data type of raw datasets.")

    # Arguments to override defaults - each is atomic
    other_args = parser.add_argument_group()
    other_args.add_argument(
        "-o", "--output", type=str, default=None, dest="output",
        help="Output file name. If None then generated as input file prefix "
             "with vds suffix.")
    other_args.add_argument(
        "-s", "--stripe_spacing", type=int, dest="stripe_spacing",
        default=SubFrameVDSGenerator.stripe_spacing,
        help="Spacing between two stripes in a module.")
    other_args.add_argument(
        "-m", "--module_spacing", type=int, dest="module_spacing",
        default=SubFrameVDSGenerator.module_spacing,
        help="Spacing between two modules.")
    other_args.add_argument(
        "-F", "--fill_value", type=int, dest="fill_value", default=0,
        help="Fill value for spacing.")
    other_args.add_argument(
        "--source_node", type=str, dest="source_node",
        default=VDSGenerator.source_node,
        help="Data node in source HDF5 files.")
    other_args.add_argument(
        "--target_node", type=str, dest="target_node",
        default=VDSGenerator.target_node, help="Data node in VDS file.")
    other_args.add_argument(
        "--mode", type=str, dest="mode", default="sub-frames",
        help="Type of raw datasets"
             "  sub-frames: ND datasets containing sub_frames of full image"
             "  frames:     1D datasets containing interspersed frames")
    other_args.add_argument(
        "-l", "--log_level", type=int, dest="log_level",
        default=VDSGenerator.log_level,
        help="Logging level (off=3, info=2, debug=1).")

    args = parser.parse_args()
    args.shape = tuple(args.shape)

    if args.empty and args.files is None:
        parser.error(
            "To make an empty VDS you must explicitly define --files for the "
            "eventual raw datasets.")
    if args.files is not None and len(args.files) < 2:
        parser.error("Must define at least two files to combine.")

    return args


def main():
    """Run program."""
    args = parse_args()

    if args.empty:
        source_metadata = dict(shape=args.shape, dtype=args.data_type)
    else:
        source_metadata = None

    if args.mode == "frames":
        gen = FrameVDSGenerator(args.path,
                                prefix=args.prefix, files=args.files,
                                output=args.output,
                                source=source_metadata,
                                source_node=args.source_node,
                                target_node=args.target_node,
                                fill_value=args.fill_value,
                                log_level=args.log_level)
    elif args.mode == "sub-frames":
        gen = SubFrameVDSGenerator(args.path,
                                   prefix=args.prefix, files=args.files,
                                   output=args.output,
                                   source=source_metadata,
                                   source_node=args.source_node,
                                   target_node=args.target_node,
                                   stripe_spacing=args.stripe_spacing,
                                   module_spacing=args.module_spacing,
                                   fill_value=args.fill_value,
                                   log_level=args.log_level)
    else:
        raise NotImplementedError("Invalid VDS mode. Must be either frames or "
                                  "sub-frames.")

    gen.generate_vds()


if __name__ == "__main__":
    sys.exit(main())
