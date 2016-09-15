import unittest

from pkg_resources import require
require("mock")
from mock import MagicMock, patch, ANY

from vdsgen import vdsgen


class ParseArgsTest(unittest.TestCase):

    @patch('argparse.ArgumentParser.parse_args')
    def test_no_args_given(self, parse_mock):
        vdsgen.parse_args()

        parse_mock.assert_called_once_with()

