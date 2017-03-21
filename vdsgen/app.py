import sys
from argparse import ArgumentParser

from vdsgenerator import VDSGenerator


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser()
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
        "--data_type", type=str, default="uint16", dest="data_type",
        help="Data type of raw datasets.")

    # Arguments to override defaults - each is atomic
    other_args = parser.add_argument_group()
    other_args.add_argument(
        "-o", "--output", type=str, default=None, dest="output",
        help="Output file name. Default is input file prefix with vds suffix.")
    other_args.add_argument(
        "-s", "--stripe_spacing", nargs="?", type=int, default=None,
        dest="stripe_spacing", help="Spacing between two stripes in a module.")
    other_args.add_argument(
        "-m", "--module_spacing", nargs="?", type=int, default=None,
        dest="module_spacing", help="Spacing between two modules.")
    other_args.add_argument(
        "--source_node", nargs="?", type=str, default=None, dest="source_node",
        help="Data node in source HDF5 files.")
    other_args.add_argument(
        "--target_node", nargs="?", type=str, default=None, dest="target_node",
        help="Data node in VDS file.")

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

    gen = VDSGenerator(args.path,
                       prefix=args.prefix, files=args.files,
                       output=args.output,
                       source=source_metadata,
                       source_node=args.source_node,
                       target_node=args.target_node,
                       stripe_spacing=args.stripe_spacing,
                       module_spacing=args.module_spacing)

    gen.generate_vds()


if __name__ == "__main__":
    sys.exit(main())
