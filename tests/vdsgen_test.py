import os
import sys
import unittest

import numpy as np

from pkg_resources import require
require("mock")
from mock import MagicMock, patch, ANY, call

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "h5py"))

from vdsgen import vdsgen

vdsgen_patch_path = "vdsgen.vdsgen"
parser_patch_path = "argparse.ArgumentParser"
h5py_patch_path = "h5py"


class ParseArgsTest(unittest.TestCase):

    @patch(parser_patch_path + '.add_argument')
    @patch(parser_patch_path + '.parse_args')
    def test_no_args_given(self, parse_mock, add_mock):
        args = vdsgen.parse_args()

        add_mock.has_calls(call("path", type=str,
                                help="Path to folder containing HDF5 files."),
                           call("prefix", type=str,
                                help="Root name of images - e.g 'stripe_' to "
                                     "combine the images 'stripe_1.hdf5', "
                                     "'stripe_2.hdf5' and 'stripe_3.hdf5' "
                                     "located at <path>."))
        parse_mock.assert_called_once_with()
        self.assertEqual(parse_mock.return_value, args)


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

        meta_data = vdsgen.grab_metadata("/test/path")

        h5file_mock.assert_called_once_with("/test/path", "r")
        self.assertEqual(expected_data, meta_data)

    @patch(vdsgen_patch_path + '.grab_metadata',
           return_value=dict(frames=3, height=256, width=2048, dtype="uint16"))
    def test_process_source_datasets_given_valid_data(self, grab_mock):
        files = ["stripe_1.h5", "stripe_2.h5"]
        expected_source = vdsgen.Source(frames=3, height=256, width=2048,
                                        dtype="uint16", datasets=files)

        source = vdsgen.process_source_datasets(files)

        grab_mock.assert_has_calls([call("stripe_1.h5"), call("stripe_2.h5")])
        self.assertEqual(expected_source, source)

    @patch(vdsgen_patch_path + '.grab_metadata',
           side_effect=[dict(frames=3, height=256, width=2048, dtype="uint16"),
                        dict(frames=4, height=256, width=2048,
                             dtype="uint16")])
    def test_process_source_datasets_given_mismatched_data(self, grab_mock):
        files = ["stripe_1.h5", "stripe_2.h5"]

        with self.assertRaises(ValueError):
            vdsgen.process_source_datasets(files)

        grab_mock.assert_has_calls([call("stripe_1.h5"), call("stripe_2.h5")])

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

        map_list = vdsgen.create_vds_maps(source, vds)

        target_mock.assert_called_once_with("/test/path", "full_frame",
                                            shape=(3, 1586, 2048))
        source_mock.assert_has_calls([call("source", "data",
                                           shape=(3, 256, 2048))]*6)
        map_mock.assert_has_calls([call(source_mock.return_value,
                                        target_mock.return_value.__getitem__.return_value,
                                        dtype="uint16")]*6)
        self.assertEqual([map_mock.return_value]*6, map_list)

    def test_create_vds_maps_system_test(self):
        mock_arrays = [np.full((256, 2048), fill) for fill in range(6)]
        source = vdsgen.Source(frames=3, height=256, width=2048,
                               dtype="uint16",
                               datasets=mock_arrays)
        vds = vdsgen.VDS(shape=(3, 1586, 2048), spacing=[10] * 5 + [0],
                         path="/test/path")

        map_list = vdsgen.create_vds_maps(source, vds)

        for idx, array in enumerate(mock_arrays[:-1]):
            np.testing.assert_array_equal(array, map_list[idx].src.path)
            self.assertEqual(idx * (256 + 10),
                             map_list[idx].target.slice_list[1].start)
            self.assertEqual((idx + 1) * (256 + 10),
                             map_list[idx].target.slice_list[1].stop)

        # Final array doesn't have gap added at the end
        np.testing.assert_array_equal(mock_arrays[-1], map_list[-1].src.path)
        self.assertEqual(5 * (256 + 10),
                         map_list[-1].target.slice_list[1].start)
        self.assertEqual((5 + 1) * (256 + 10) - 10,
                         map_list[-1].target.slice_list[1].stop)


class MainTest(unittest.TestCase):

    file_mock = MagicMock()

    @patch(h5py_patch_path + '.File', return_value=file_mock)
    @patch(vdsgen_patch_path + '.create_vds_maps')
    @patch(vdsgen_patch_path + '.construct_vds_metadata')
    @patch(vdsgen_patch_path + '.process_source_datasets')
    @patch(vdsgen_patch_path + '.construct_vds_name',
           return_value="stripe_vds.h5")
    @patch(vdsgen_patch_path + '.find_files',
           return_value=["stripe_1.hdf5", "stripe_2.hdf5", "stripe_3.hdf5"])
    def test_generate_vds(self, find_mock, gen_mock, process_mock,
                          construct_mock, create_mock, h5file_mock):
        vds_file_mock = self.file_mock.__enter__.return_value

        vdsgen.generate_vds("/test/path", "stripe_")

        find_mock.assert_called_once_with("/test/path", "stripe_")
        gen_mock.assert_called_once_with("stripe_", find_mock.return_value)
        process_mock.assert_called_once_with(find_mock.return_value)
        construct_mock.assert_called_once_with(process_mock.return_value,
                                               "/test/path/stripe_vds.h5")
        create_mock.assert_called_once_with(process_mock.return_value,
                                            construct_mock.return_value)
        h5file_mock.assert_called_once_with("/test/path/stripe_vds.h5", "w",
                                            libver="latest")
        vds_file_mock.create_virtual_dataset.assert_called_once_with(
            VMlist=create_mock.return_value, fill_value=0x1)

    @patch(vdsgen_patch_path + '.generate_vds')
    @patch(vdsgen_patch_path + '.parse_args',
           return_value=MagicMock(path="/test/path", prefix="stripe_"))
    def test_main(self, parse_mock, generate_mock):
        args_mock = parse_mock.return_value

        vdsgen.main()

        parse_mock.assert_called_once_with()
        generate_mock.assert_called_once_with(args_mock.path, args_mock.prefix)
        source = vdsgen.Source(frames=3, height=256, width=2048,
                               dtype="uint16", datasets=[""]*6)
