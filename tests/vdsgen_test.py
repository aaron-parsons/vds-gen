import unittest

from pkg_resources import require
require("mock")
from mock import MagicMock, patch, ANY, call
vdsgen_patch_path = "vdsgen.vdsgen"
parser_patch_path = "argparse.ArgumentParser"
h5py_patch_path = "h5py"

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "h5py"))

from vdsgen import vdsgen


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

        args = vdsgen.parse_args()

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
            [call("--frames", type=int, default=1, dest="frames",
                  help="Number of frames to combine into VDS."),
             call("--height", type=int, default=256, dest="height",
                  help="Height of raw datasets."),
             call("--width", type=int, default=1024, dest="width",
                  help="Width of raw datasets."),
             call("--data_type", type=str, default="uint16", dest="data_type",
                  help="Data type of raw datasets.")]
        )

        add_exclusive_group_mock.assert_called_with(required=True)
        exclusive_group_mock.add_argument.assert_has_calls(
            [call("-p", "--prefix", type=str, default=None, dest="prefix",
                  help="Prefix of files - e.g 'stripe_' to combine the images "
                       "'stripe_1.hdf5', 'stripe_2.hdf5' and 'stripe_3.hdf5' "
                       "located at <path>."),
             call("-f", "--files", nargs="*", type=str, default=None,
                  dest="files",
                  help="Manually define files to combine.")]
        )

        parse_mock.assert_called_once_with()
        self.assertEqual(parse_mock.return_value, args)

    @patch(parser_patch_path + '.error')
    @patch(parser_patch_path + '.parse_args',
           return_value=MagicMock(empty=True, files=None))
    def test_empty_and_not_files_then_error(self, parse_mock, error_mock):

        vdsgen.parse_args()

        error_mock.assert_called_once_with(
            "To make an empty VDS you must explicitly define --files for the "
            "eventual raw datasets.")

    @patch(parser_patch_path + '.error')
    @patch(parser_patch_path + '.parse_args',
           return_value=MagicMock(empty=True, files=["file"]))
    def test_only_one_file_then_error(self, parse_mock, error_mock):
        vdsgen.parse_args()

        error_mock.assert_called_once_with(
            "Must define at least two files to combine.")


class FindFilesTest(unittest.TestCase):

    @patch('os.listdir',
           return_value=["stripe_1.h5", "stripe_2.h5", "stripe_3.h5",
                         "stripe_4.h5", "stripe_5.h5", "stripe_6.h5"])
    def test_given_files_then_return(self, _):
        expected_files = ["/test/path/stripe_1.h5", "/test/path/stripe_2.h5",
                          "/test/path/stripe_3.h5", "/test/path/stripe_4.h5",
                          "/test/path/stripe_5.h5", "/test/path/stripe_6.h5"]

        files = vdsgen.find_files("/test/path", "stripe_")

        self.assertEqual(expected_files, files)

    @patch('os.listdir',
           return_value=["stripe_4.h5", "stripe_1.h5", "stripe_6.h5",
                         "stripe_3.h5", "stripe_2.h5", "stripe_5.h5"])
    def test_given_files_out_of_order_then_return(self, _):
        expected_files = ["/test/path/stripe_1.h5", "/test/path/stripe_2.h5",
                          "/test/path/stripe_3.h5", "/test/path/stripe_4.h5",
                          "/test/path/stripe_5.h5", "/test/path/stripe_6.h5"]

        files = vdsgen.find_files("/test/path", "stripe_")

        self.assertEqual(expected_files, files)

    @patch('os.listdir', return_value=["stripe_1.h5"])
    def test_given_one_file_then_error(self, _):

        with self.assertRaises(IOError):
            vdsgen.find_files("/test/path", "stripe_")

    @patch('os.listdir', return_value=[])
    def test_given_no_files_then_error(self, _):

        with self.assertRaises(IOError):
            vdsgen.find_files("/test/path", "stripe_")


class SimpleFunctionsTest(unittest.TestCase):

    def test_generate_vds_name(self):
        expected_name = "stripe_vds.h5"
        files = ["stripe_1.h5", "stripe_2.h5", "stripe_3.h5",
                 "stripe_4.h5", "stripe_5.h5", "stripe_6.h5"]

        vds_name = vdsgen.construct_vds_name("stripe_", files)

        self.assertEqual(expected_name, vds_name)

    mock_data = dict(data=MagicMock(shape=(3, 256, 2048), dtype="uint16"))

    @patch(h5py_patch_path + '.File', return_value=mock_data)
    def test_grab_metadata(self, h5file_mock):
        expected_data = dict(frames=3, height=256, width=2048, dtype="uint16")

        meta_data = vdsgen.grab_metadata("/test/path", "data")

        h5file_mock.assert_called_once_with("/test/path", "r")
        self.assertEqual(expected_data, meta_data)

    @patch(vdsgen_patch_path + '.grab_metadata',
           return_value=dict(frames=3, height=256, width=2048, dtype="uint16"))
    def test_process_source_datasets_given_valid_data(self, grab_mock):
        files = ["stripe_1.h5", "stripe_2.h5"]
        expected_source = vdsgen.Source(frames=3, height=256, width=2048,
                                        dtype="uint16", datasets=files)

        source = vdsgen.process_source_datasets(files, "data")

        grab_mock.assert_has_calls([call("stripe_1.h5", "data"),
                                    call("stripe_2.h5", "data")])
        self.assertEqual(expected_source, source)

    @patch(vdsgen_patch_path + '.grab_metadata',
           side_effect=[dict(frames=3, height=256, width=2048, dtype="uint16"),
                        dict(frames=4, height=256, width=2048,
                             dtype="uint16")])
    def test_process_source_datasets_given_mismatched_data(self, grab_mock):
        files = ["stripe_1.h5", "stripe_2.h5"]

        with self.assertRaises(ValueError):
            vdsgen.process_source_datasets(files, "data")

        grab_mock.assert_has_calls([call("stripe_1.h5", "data"),
                                    call("stripe_2.h5", "data")])

    def test_construct_vds_metadata(self):
        source = vdsgen.Source(frames=3, height=256, width=2048,
                               dtype="uint16", datasets=[""]*6)
        expected_vds = vdsgen.VDS(shape=(3, 1586, 2048),
                                  spacing=[10] * 5 + [0],
                                  path="/test/path")

        vds = vdsgen.construct_vds_metadata(source, "/test/path")

        self.assertEqual(expected_vds, vds)

    @patch(h5py_patch_path + '.VirtualMap')
    @patch(h5py_patch_path + '.VirtualSource')
    @patch(h5py_patch_path + '.VirtualTarget')
    def test_create_vds_maps(self, target_mock, source_mock, map_mock):
        source = vdsgen.Source(frames=3, height=256, width=2048,
                               dtype="uint16", datasets=["source"]*6)
        vds = vdsgen.VDS(shape=(3, 1586, 2048), spacing=[10] * 5 + [0],
                         path="/test/path")

        map_list = vdsgen.create_vds_maps(source, vds, "data", "full_frame")

        target_mock.assert_called_once_with("/test/path", "full_frame",
                                            shape=(3, 1586, 2048))
        source_mock.assert_has_calls([call("source", "data",
                                           shape=(3, 256, 2048))]*6)
        # TODO: Improve this assert by passing numpy arrays to check slicing
        map_mock.assert_has_calls([call(source_mock.return_value,
                                        target_mock.return_value.__getitem__.return_value,
                                        dtype="uint16")]*6)
        self.assertEqual([map_mock.return_value]*6, map_list)


class ValidateNodeTest(unittest.TestCase):

    def setUp(self):
        self.file_mock = MagicMock()

    def test_validate_node_creates(self):
        self.file_mock.get.return_value = None

        vdsgen.validate_node(self.file_mock, "entry/detector/detector1")

        self.file_mock.create_group.assert_called_once_with("entry/detector")

    def test_validate_node_exists_then_no_op(self):
        self.file_mock.get.return_value = "Group"

        vdsgen.validate_node(self.file_mock, "entry/detector/detector1")

        self.file_mock.create_group.assert_not_called()

    def test_validate_node_invalid_then_error(self):

        with self.assertRaises(ValueError):
            vdsgen.validate_node(self.file_mock, "/entry/detector/detector1")
        with self.assertRaises(ValueError):
            vdsgen.validate_node(self.file_mock, "entry/detector/detector1/")
        with self.assertRaises(ValueError):
            vdsgen.validate_node(self.file_mock, "/entry/detector/detector1/")


class MainTest(unittest.TestCase):

    file_mock = MagicMock()
    file_mock_2 = MagicMock()

    @patch('os.path.isfile', side_effect=[False, True, True, True])
    @patch(h5py_patch_path + '.File', return_value=file_mock)
    @patch(vdsgen_patch_path + '.create_vds_maps')
    @patch(vdsgen_patch_path + '.construct_vds_metadata')
    @patch(vdsgen_patch_path + '.process_source_datasets')
    @patch(vdsgen_patch_path + '.construct_vds_name',
           return_value="stripe_vds.h5")
    @patch(vdsgen_patch_path + '.find_files',
           return_value=["stripe_1.hdf5", "stripe_2.hdf5", "stripe_3.hdf5"])
    def test_generate_vds_defaults(self, find_mock, gen_mock, process_mock,
                                   construct_mock, create_mock, h5file_mock,
                                   isfile_mock):
        self.file_mock.reset_mock()
        vds_file_mock = self.file_mock.__enter__.return_value

        vdsgen.generate_vds("/test/path", prefix="stripe_")

        find_mock.assert_called_once_with("/test/path", "stripe_")
        gen_mock.assert_called_once_with("stripe_", find_mock.return_value)
        process_mock.assert_called_once_with(find_mock.return_value, "data")
        construct_mock.assert_called_once_with(process_mock.return_value,
                                               "/test/path/stripe_vds.h5",
                                               None, None)
        create_mock.assert_called_once_with(process_mock.return_value,
                                            construct_mock.return_value,
                                            "data", "full_frame")
        h5file_mock.assert_called_once_with("/test/path/stripe_vds.h5", "w",
                                            libver="latest")
        vds_file_mock.create_virtual_dataset.assert_called_once_with(
            VMlist=create_mock.return_value, fill_value=0x1)

    @patch('os.path.isfile', return_value=True)
    @patch(h5py_patch_path + '.File', side_effect=[file_mock, file_mock_2])
    @patch(vdsgen_patch_path + '.create_vds_maps')
    @patch(vdsgen_patch_path + '.construct_vds_metadata')
    def test_generate_vds_given_args(self, metadata_mock,
                                     create_mock, h5file_mock, isfile_mock):
        self.file_mock.reset_mock()
        self.file_mock_2.reset_mock()
        self.file_mock.__enter__.return_value.get.return_value = None
        vds_file_mock = self.file_mock_2.__enter__.return_value
        files = ["stripe_1.h5", "stripe_2.h5"]
        file_paths = ["/test/path/" + file_ for file_ in files]
        source_dict = dict(frames=3, height=256, width=1024, dtype="int16")
        source = vdsgen.Source(frames=3, height=256, width=1024, dtype="int16",
                               datasets=file_paths)

        vdsgen.generate_vds("/test/path", files=files, output="vds.h5",
                            source=source_dict,
                            source_node="data", target_node="full_frame",
                            stripe_spacing=3, module_spacing=127)

        metadata_mock.assert_called_once_with(source,
                                              "/test/path/vds.h5",
                                              3, 127)
        create_mock.assert_called_once_with(source,
                                            metadata_mock.return_value,
                                            "data", "full_frame")
        h5file_mock.assert_has_calls([
            call("/test/path/vds.h5", "r", libver="latest"),
            call("/test/path/vds.h5", "w", libver="latest")])
        vds_file_mock.create_virtual_dataset.assert_called_once_with(
            VMlist=create_mock.return_value, fill_value=0x1)

    def test_generate_vds_prefix_and_files_then_error(self):

        with self.assertRaises(ValueError):
            vdsgen.generate_vds("/test/path", "stripe_", ["file1", "file2"])

    @patch('os.path.isfile', return_value=False)
    @patch(vdsgen_patch_path + '.construct_vds_name',
           return_value="stripe_vds.h5")
    def test_generate_vds_no_source_or_files_then_error(self, construct_mock,
                                                        isfile_mock):

        with self.assertRaises(IOError) as e:
            vdsgen.generate_vds("/test/path", files=["file1", "file2"])
        self.assertEqual("File /test/path/file1 does not exist. To create VDS "
                         "from raw files that haven't been created yet, "
                         "source must be provided.",
                         e.exception.message)

    @patch('os.path.isfile', return_value=True)
    @patch(h5py_patch_path + '.File', return_value=file_mock)
    def test_generate_vds_target_node_exists_then_error(self, h5_file_mock,
                                                        isfile_mock):
        self.file_mock.reset_mock()
        self.file_mock.__enter__.return_value.get.return_value = MagicMock()

        with self.assertRaises(IOError) as e:
            vdsgen.generate_vds("/test/path", files=["file1", "file2"],
                                output="vds")
        self.assertEqual("VDS /test/path/vds already has an entry for node "
                         "full_frame",
                         e.exception.message)

    @patch(vdsgen_patch_path + '.generate_vds')
    @patch(vdsgen_patch_path + '.parse_args',
           return_value=MagicMock(
               path="/test/path", prefix="stripe_", empty=True,
               files=["file1.hdf5", "file2.hdf5"], output="vds",
               frames=3, height=256, width=2048, data_type="int16",
               source_node="data", target_node="full_frame",
               stripe_spacing=3, module_spacing=127))
    def test_main_empty(self, parse_mock, generate_mock):
        args_mock = parse_mock.return_value

        vdsgen.main()

        parse_mock.assert_called_once_with()
        generate_mock.assert_called_once_with(
            args_mock.path,
            prefix=args_mock.prefix, output="vds", files=args_mock.files,
            source=dict(frames=args_mock.frames, height=args_mock.height,
                        width=args_mock.width, dtype=args_mock.data_type),
            source_node=args_mock.source_node,
            target_node=args_mock.target_node,
            stripe_spacing=args_mock.stripe_spacing,
            module_spacing=args_mock.module_spacing)

    @patch(vdsgen_patch_path + '.generate_vds')
    @patch(vdsgen_patch_path + '.parse_args',
           return_value=MagicMock(
               path="/test/path", prefix="stripe_", empty=False,
               files=["file1.hdf5", "file2.hdf5"], output="vds",
               frames=3, height=256, width=2048, data_type="int16",
               source_node="data", target_node="full_frame",
               stripe_spacing=3, module_spacing=127))
    def test_main_not_empty(self, parse_mock, generate_mock):
        args_mock = parse_mock.return_value

        vdsgen.main()

        parse_mock.assert_called_once_with()
        generate_mock.assert_called_once_with(
            args_mock.path,
            prefix=args_mock.prefix, output="vds", files=args_mock.files,
            source=None,
            source_node=args_mock.source_node,
            stripe_spacing=args_mock.stripe_spacing,
            target_node=args_mock.target_node,
            module_spacing=args_mock.module_spacing)
