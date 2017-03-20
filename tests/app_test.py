import unittest
from pkg_resources import require

require("mock")
from mock import MagicMock, patch, call

from vdsgen import app

parser_patch_path = "argparse.ArgumentParser"
app_patch_path = "vdsgen.app"
VDSGenerator_patch_path = app_patch_path + ".VDSGenerator"


class ParseArgsTest(unittest.TestCase):

    @patch(parser_patch_path + '.add_mutually_exclusive_group')
    @patch(parser_patch_path + '.add_argument_group')
    @patch(parser_patch_path + '.add_argument')
    @patch(parser_patch_path + '.parse_args',
           return_value=MagicMock(empty=False, files=None))
    def test_parser(self, parse_mock, add_mock, add_group_mock,
                    add_exclusive_group_mock):
        group_mock = add_group_mock.return_value
        exclusive_group_mock = add_exclusive_group_mock.return_value

        args = app.parse_args()

        add_mock.assert_has_calls(
            [call("path", type=str,
                  help="Root folder to create VDS in. Also where source "
                       "files are searched for if --prefix given."),
             call("-o", "--output", type=str, default=None, dest="output",
                  help="Output file name. Default is input file prefix with "
                       "vds suffix."),
             call("-e", "--empty", action="store_true", dest="empty",
                  help="Make empty VDS pointing to datasets "
                       "that don't exist, yet."),
             call("-s", "--stripe_spacing", nargs="?", type=int, default=None,
                  dest="stripe_spacing",
                  help="Spacing between two stripes in a module."),
             call("-m", "--module_spacing", nargs="?", type=int, default=None,
                  dest="module_spacing",
                  help="Spacing between two modules."),
             call("--source_node", nargs="?", type=str, default=None,
                  dest="source_node",
                  help="Data node in source HDF5 files."),
             call("--target_node", nargs="?", type=str, default=None,
                  dest="target_node",
                  help="Data node in VDS file.")])

        add_group_mock.assert_called_with()
        group_mock.add_argument.assert_has_calls(
            [call("--shape", type=int, nargs="*", default=[1, 256, 2048],
                  dest="shape",
                  help="Shape of dataset - 'frames height width', where "
                       "frames is N dimensional."),
             call("--data_type", type=str, default="uint16", dest="data_type",
                  help="Data type of raw datasets.")])

        add_exclusive_group_mock.assert_called_with(required=True)
        exclusive_group_mock.add_argument.assert_has_calls(
            [call("-p", "--prefix", type=str, default=None, dest="prefix",
                  help="Prefix of files - e.g 'stripe_' to combine the images "
                       "'stripe_1.hdf5', 'stripe_2.hdf5' and 'stripe_3.hdf5' "
                       "located at <path>."),
             call("-f", "--files", nargs="*", type=str, default=None,
                  dest="files",
                  help="Manually define files to combine.")])

        parse_mock.assert_called_once_with()
        self.assertEqual(parse_mock.return_value, args)

    @patch(parser_patch_path + '.error')
    @patch(parser_patch_path + '.parse_args',
           return_value=MagicMock(empty=True, files=None))
    def test_empty_and_not_files_then_error(self, parse_mock, error_mock):

        app.parse_args()

        error_mock.assert_called_once_with(
            "To make an empty VDS you must explicitly define --files for the "
            "eventual raw datasets.")

    @patch(parser_patch_path + '.error')
    @patch(parser_patch_path + '.parse_args',
           return_value=MagicMock(empty=True, files=["file"]))
    def test_only_one_file_then_error(self, parse_mock, error_mock):

        app.parse_args()

        error_mock.assert_called_once_with(
            "Must define at least two files to combine.")


class MainTest(unittest.TestCase):
    @patch(VDSGenerator_patch_path)
    @patch(app_patch_path + '.parse_args',
           return_value=MagicMock(
               path="/test/path", prefix="stripe_", empty=True,
               files=["file1.hdf5", "file2.hdf5"], output="vds",
               shape=[3, 256, 2048], data_type="int16",
               source_node="data", target_node="full_frame",
               stripe_spacing=3, module_spacing=127))
    def test_main_empty(self, parse_mock, init_mock):
        gen_mock = init_mock.return_value
        args_mock = parse_mock.return_value

        app.main()

        parse_mock.assert_called_once_with()
        init_mock.assert_called_once_with(
            args_mock.path,
            prefix=args_mock.prefix, files=args_mock.files,
            output=args_mock.output,
            source=dict(shape=args_mock.shape, dtype=args_mock.data_type),
            source_node=args_mock.source_node,
            target_node=args_mock.target_node,
            stripe_spacing=args_mock.stripe_spacing,
            module_spacing=args_mock.module_spacing)

        gen_mock.generate_vds.assert_called_once_with()

    @patch(VDSGenerator_patch_path)
    @patch(app_patch_path + '.parse_args',
           return_value=MagicMock(
               path="/test/path", prefix="stripe_", empty=False,
               files=["file1.hdf5", "file2.hdf5"], output="vds",
               frames=3, height=256, width=2048, data_type="int16",
               source_node="data", target_node="full_frame",
               stripe_spacing=3, module_spacing=127))
    def test_main_not_empty(self, parse_mock, generate_mock):
        args_mock = parse_mock.return_value

        app.main()

        parse_mock.assert_called_once_with()
        generate_mock.assert_called_once_with(
            args_mock.path,
            prefix=args_mock.prefix, output="vds", files=args_mock.files,
            source=None,
            source_node=args_mock.source_node,
            stripe_spacing=args_mock.stripe_spacing,
            target_node=args_mock.target_node,
            module_spacing=args_mock.module_spacing)