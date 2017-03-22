import os
import sys
import unittest

from pkg_resources import require
require("mock")
from mock import MagicMock, patch, call

from vdsgen import vdsgenerator
from vdsgen.vdsgenerator import VDSGenerator

vdsgen_patch_path = "vdsgen.vdsgenerator"
VDSGenerator_patch_path = vdsgen_patch_path + ".VDSGenerator"
h5py_patch_path = "h5py"

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "h5py"))


class VDSGeneratorTester(VDSGenerator):

    """A version of VDSGenerator without initialisation.

    For testing single methods of the class. Must have required attributes
    passed before calling testee function.

    """

    def __init__(self, **kwargs):
        for attribute, value in kwargs.items():
            self.__setattr__(attribute, value)


class VDSGeneratorInitTest(unittest.TestCase):

    @patch('os.path.isfile', return_value=True)
    @patch(VDSGenerator_patch_path + '.process_source_datasets')
    @patch(VDSGenerator_patch_path + '.construct_vds_name',
           return_value="stripe_vds.hdf5")
    @patch(VDSGenerator_patch_path + '.find_files',
           return_value=["/test/path/stripe_1.hdf5",
                         "/test/path/stripe_2.hdf5",
                         "/test/path/stripe_3.hdf5"])
    def test_generate_vds_defaults(self, find_mock, construct_mock,
                                   process_mock, isfile_mock):
        expected_files = ["stripe_1.hdf5", "stripe_2.hdf5", "stripe_3.hdf5"]

        gen = VDSGenerator("/test/path", prefix="stripe_")

        find_mock.assert_called_once_with()
        construct_mock.assert_called_once_with(expected_files)
        process_mock.assert_called_once_with()

        self.assertEqual("/test/path", gen.path)
        self.assertEqual("stripe_", gen.prefix)
        self.assertEqual("stripe_vds.hdf5", gen.name)
        self.assertEqual(find_mock.return_value, gen.datasets)
        self.assertEqual(process_mock.return_value, gen.source_metadata)
        self.assertEqual("data", gen.source_node)
        self.assertEqual("full_frame", gen.target_node)
        self.assertEqual(10, gen.stripe_spacing)
        self.assertEqual(10, gen.module_spacing)
        self.assertEqual(gen.CREATE, gen.mode)

    def test_generate_vds_given_args(self):
        files = ["stripe_1.h5", "stripe_2.h5"]
        file_paths = ["/test/path/" + file_ for file_ in files]
        source_dict = dict(shape=(3, 256, 2048), dtype="int16")
        source = vdsgenerator.Source(frames=(3,), height=256, width=2048,
                                     dtype="int16")

        gen = VDSGenerator("/test/path",
                           files=files,
                           output="vds.hdf5",
                           source=source_dict,
                           source_node="entry/data/data",
                           target_node="entry/detector/detector1",
                           stripe_spacing=3, module_spacing=127)

        self.assertEqual("/test/path", gen.path)
        self.assertEqual("stripe_", gen.prefix)
        self.assertEqual("vds.hdf5", gen.name)
        self.assertEqual(file_paths, gen.datasets)
        self.assertEqual(source, gen.source_metadata)
        self.assertEqual("entry/data/data", gen.source_node)
        self.assertEqual("entry/detector/detector1", gen.target_node)
        self.assertEqual(3, gen.stripe_spacing)
        self.assertEqual(127, gen.module_spacing)
        self.assertEqual(gen.CREATE, gen.mode)

    def test_generate_vds_prefix_and_files_then_error(self):
        files = ["stripe_1.h5", "stripe_2.h5"]
        source_dict = dict(frames=3, height=256, width=2048, dtype="int16")

        with self.assertRaises(ValueError):
            VDSGenerator("/test/path",
                         prefix="stripe_", files=files,
                         output="vds.hdf5",
                         source=source_dict,
                         source_node="entry/data/data",
                         target_node="entry/detector/detector1",
                         stripe_spacing=3, module_spacing=127)

    @patch('os.path.isfile', return_value=False)
    def test_generate_vds_no_source_or_files_then_error(self, _):

        with self.assertRaises(IOError) as e:
            VDSGenerator("/test/path",
                         files=["file1", "file2"],
                         output="vds.hdf5")

        self.assertEqual("File /test/path/file1 does not exist. To create VDS "
                         "from raw files that haven't been created yet, "
                         "source must be provided.",
                         e.exception.message)


class FindFilesTest(unittest.TestCase):

    def setUp(self):
        self.gen = VDSGeneratorTester(path="/test/path", prefix="stripe_")

    @patch('os.listdir',
           return_value=["stripe_1.h5", "stripe_2.h5", "stripe_3.h5",
                         "stripe_4.h5", "stripe_5.h5", "stripe_6.h5"])
    def test_given_files_then_return(self, _):
        expected_files = ["/test/path/stripe_1.h5", "/test/path/stripe_2.h5",
                          "/test/path/stripe_3.h5", "/test/path/stripe_4.h5",
                          "/test/path/stripe_5.h5", "/test/path/stripe_6.h5"]

        files = self.gen.find_files()

        self.assertEqual(expected_files, files)

    @patch('os.listdir',
           return_value=["stripe_4.h5", "stripe_1.h5", "stripe_6.h5",
                         "stripe_3.h5", "stripe_2.h5", "stripe_5.h5"])
    def test_given_files_out_of_order_then_return(self, _):
        expected_files = ["/test/path/stripe_1.h5", "/test/path/stripe_2.h5",
                          "/test/path/stripe_3.h5", "/test/path/stripe_4.h5",
                          "/test/path/stripe_5.h5", "/test/path/stripe_6.h5"]

        files = self.gen.find_files()

        self.assertEqual(expected_files, files)

    @patch('os.listdir', return_value=["stripe_1.h5"])
    def test_given_one_file_then_error(self, _):

        with self.assertRaises(IOError):
            self.gen.find_files()

    @patch('os.listdir', return_value=[])
    def test_given_no_files_then_error(self, _):

        with self.assertRaises(IOError):
            self.gen.find_files()


class SimpleFunctionsTest(unittest.TestCase):

    def test_generate_vds_name(self):
        gen = VDSGeneratorTester(prefix="stripe_")
        expected_name = "stripe_vds.h5"
        files = ["stripe_1.h5", "stripe_2.h5", "stripe_3.h5",
                 "stripe_4.h5", "stripe_5.h5", "stripe_6.h5"]

        vds_name = gen.construct_vds_name(files)

        self.assertEqual(expected_name, vds_name)

    mock_data = dict(data=MagicMock(shape=(3, 256, 2048), dtype="uint16"))

    @patch(h5py_patch_path + '.File', return_value=mock_data)
    def test_grab_metadata(self, h5file_mock):
        gen = VDSGeneratorTester(source_node="data")
        expected_data = dict(frames=(3,), height=256, width=2048, dtype="uint16")

        meta_data = gen.grab_metadata("/test/path/stripe.hdf5")

        h5file_mock.assert_called_once_with("/test/path/stripe.hdf5", "r")
        self.assertEqual(expected_data, meta_data)

    @patch(VDSGenerator_patch_path + '.grab_metadata',
           return_value=dict(frames=(3,), height=256, width=2048, dtype="uint16"))
    def test_process_source_datasets_given_valid_data(self, grab_mock):
        gen = VDSGeneratorTester(datasets=["stripe_1.h5", "stripe_2.h5"])
        expected_source = vdsgenerator.Source(frames=(3,), height=256,
                                              width=2048,
                                              dtype="uint16")

        source = gen.process_source_datasets()

        grab_mock.assert_has_calls([call("stripe_1.h5"), call("stripe_2.h5")])
        self.assertEqual(expected_source, source)

    @patch(VDSGenerator_patch_path + '.grab_metadata',
           side_effect=[dict(frames=3, height=256, width=2048, dtype="uint16"),
                        dict(frames=4, height=256, width=2048,
                             dtype="uint16")])
    def test_process_source_datasets_given_mismatched_data(self, grab_mock):
        gen = VDSGeneratorTester(datasets=["stripe_1.h5", "stripe_2.h5"])

        with self.assertRaises(ValueError):
            gen.process_source_datasets()

        grab_mock.assert_has_calls([call("stripe_1.h5"), call("stripe_2.h5")])

    def test_construct_vds_metadata(self):
        gen = VDSGeneratorTester(datasets=[""] * 6, stripe_spacing=10,
                                 module_spacing=100)
        source = vdsgenerator.Source(frames=(3,), height=256, width=2048,
                                     dtype="uint16")
        expected_vds = vdsgenerator.VDS(shape=(3, 1766, 2048),
                                        spacing=[10, 100, 10, 100, 10, 0])

        vds = gen.construct_vds_metadata(source)

        self.assertEqual(expected_vds, vds)

    @patch(h5py_patch_path + '.VirtualMap')
    @patch(h5py_patch_path + '.VirtualSource')
    @patch(h5py_patch_path + '.VirtualTarget')
    def test_create_vds_maps(self, target_mock, source_mock, map_mock):
        gen = VDSGeneratorTester(output_file="/test/path/vds.hdf5",
                                 stripe_spacing=10, module_spacing=100,
                                 target_node="full_frame", source_node="data",
                                 datasets=["source"] * 6, name="vds.hdf5")
        source = vdsgenerator.Source(frames=(3,), height=256, width=2048,
                                     dtype="uint16")
        vds = vdsgenerator.VDS(shape=(3, 1586, 2048), spacing=[10] * 5 + [0])

        map_list = gen.create_vds_maps(source, vds)

        target_mock.assert_called_once_with("/test/path/vds.hdf5",
                                            "full_frame",
                                            shape=(3, 1586, 2048))
        source_mock.assert_has_calls([call("source", "data",
                                           shape=(3, 256, 2048))] * 6)
        # TODO: Improve this assert by passing numpy arrays to check slicing
        map_mock.assert_has_calls([
            call(source_mock.return_value,
                 target_mock.return_value.__getitem__.return_value,
                 dtype="uint16")]*6)
        self.assertEqual([map_mock.return_value]*6, map_list)


class ValidateNodeTest(unittest.TestCase):

    def setUp(self):
        self.file_mock = MagicMock()

    def test_validate_node_creates(self):
        gen = VDSGeneratorTester(target_node="/entry/detector/detector1")
        self.file_mock.get.return_value = None

        gen.validate_node(self.file_mock)

        self.file_mock.create_group.assert_called_once_with("/entry/detector")

    def test_validate_node_exists_then_no_op(self):
        gen = VDSGeneratorTester(target_node="entry/detector/detector1")
        self.file_mock.get.return_value = "Group"

        gen.validate_node(self.file_mock)

        self.file_mock.create_group.assert_not_called()

    def test_validate_node_trailing_slash_then_removed(self):
        gen = VDSGeneratorTester(target_node="/entry/detector/detector1//")
        self.file_mock.get.return_value = None

        gen.validate_node(self.file_mock)

        self.file_mock.create_group.assert_called_once_with("/entry/detector")


class GenerateVDSTest(unittest.TestCase):

    file_mock = MagicMock()

    @patch('os.path.isfile', return_value=False)
    @patch(VDSGenerator_patch_path + '.validate_node')
    @patch(h5py_patch_path + '.File', return_value=file_mock)
    @patch(VDSGenerator_patch_path + '.create_vds_maps')
    @patch(VDSGenerator_patch_path + '.construct_vds_metadata')
    def test_generate_vds_create(self, construct_mock, create_mock,
                                 h5file_mock, validate_mock, isfile_mock):
        source_mock = MagicMock()
        gen = VDSGeneratorTester(path="/test/path", prefix="stripe_",
                                 output_file="/test/path/vds.hdf5",
                                 name="vds.hdf5",
                                 target_node="full_frame", source_node="data",
                                 datasets=["stripe_1.hdf5", "stripe_2.hdf5",
                                           "stripe_3.hdf5"],
                                 source_metadata=source_mock)
        self.file_mock.reset_mock()
        vds_file_mock = self.file_mock.__enter__.return_value
        vds_file_mock.get.return_value = None

        gen.generate_vds()

        isfile_mock.assert_called_once_with("/test/path/vds.hdf5")
        construct_mock.assert_called_once_with(source_mock)
        create_mock.assert_called_once_with(source_mock,
                                            construct_mock.return_value)
        validate_mock.assert_called_once_with(vds_file_mock)
        h5file_mock.assert_called_once_with(
            "/test/path/vds.hdf5", "w", libver="latest")
        vds_file_mock.create_virtual_dataset.assert_called_once_with(
            VMlist=create_mock.return_value, fillvalue=0x1)

    @patch('os.path.isfile', return_value=True)
    @patch(VDSGenerator_patch_path + '.validate_node')
    @patch(h5py_patch_path + '.File', return_value=file_mock)
    @patch(VDSGenerator_patch_path + '.create_vds_maps')
    @patch(VDSGenerator_patch_path + '.construct_vds_metadata')
    def test_generate_vds_append(self, construct_mock, create_mock,
                                 h5file_mock, validate_mock, isfile_mock):
        source_mock = MagicMock()
        gen = VDSGeneratorTester(path="/test/path", prefix="stripe_",
                                 output_file="/test/path/vds.hdf5",
                                 name="vds.hdf5",
                                 target_node="full_frame", source_node="data",
                                 datasets=["stripe_1.hdf5", "stripe_2.hdf5",
                                           "stripe_3.hdf5"],
                                 source_metadata=source_mock)
        self.file_mock.reset_mock()
        vds_file_mock = self.file_mock.__enter__.return_value
        vds_file_mock.get.return_value = None

        gen.generate_vds()

        isfile_mock.assert_called_once_with("/test/path/vds.hdf5")
        construct_mock.assert_called_once_with(source_mock)
        create_mock.assert_called_once_with(source_mock,
                                            construct_mock.return_value)
        validate_mock.assert_called_once_with(vds_file_mock)
        h5file_mock.assert_has_calls([
            call("/test/path/vds.hdf5", "r", libver="latest"),
            call("/test/path/vds.hdf5", "a", libver="latest")])
        vds_file_mock.create_virtual_dataset.assert_called_once_with(
            VMlist=create_mock.return_value, fillvalue=0x1)

    @patch('os.path.isfile', return_value=True)
    @patch(h5py_patch_path + '.File', return_value=file_mock)
    def test_generate_vds_node_exists_then_error(self, h5file_mock,
                                                 isfile_mock):
        source_mock = MagicMock()
        gen = VDSGeneratorTester(path="/test/path", prefix="stripe_",
                                 output_file="/test/path/vds.hdf5",
                                 name="vds.hdf5",
                                 target_node="full_frame", source_node="data",
                                 datasets=["stripe_1.hdf5", "stripe_2.hdf5",
                                           "stripe_3.hdf5"],
                                 source_metadata=source_mock)
        self.file_mock.reset_mock()
        vds_file_mock = self.file_mock.__enter__.return_value
        vds_file_mock.get.return_value = "Group"

        with self.assertRaises(IOError):
            gen.generate_vds()
