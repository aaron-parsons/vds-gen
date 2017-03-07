from time import sleep
import subprocess

from pkg_resources import require
require("cothread")
from cothread.catools import caget, caput, DBR_CHAR_STR

# Configuration ###############################################################

file_path = "/dls/i14/data/2017/cm16755-1/tmp/excalibur"
file_name = "stripe"
num_frames = 10
acquire_time = 10
stripe_height = 256
stripe_width = 2069
data_type = "int16"
reset_file_num = False

# Shortcuts ###################################################################

ALL_FEMS = "CONFIG"
FEMS = [1, 2, 3, 4, 5, 6]

IOC_ROOT = "BL14I-EA-EXCBR-01:{}:"

ACQUIRE = IOC_ROOT + "ACQUIRE:Acquire"
ACQ_TIME = IOC_ROOT + "ACQUIRE:AcquireTime"
ACQ_NUM = IOC_ROOT + "ACQUIRE:NumImages"

ACQ_MODE = IOC_ROOT + "ACQUIRE:ImageMode"
SINGLE_MODE = 0
MULTIPLE_MODE = 1

HDF_INPUT = IOC_ROOT + "HDF:NDArrayPort"
HDF_ENABLE = IOC_ROOT + "HDF:EnableCallbacks"
HDF_NUM = IOC_ROOT + "HDF:NumCapture"
HDF_FILE_NUM = IOC_ROOT + "HDF:FileNumber"
HDF_TEMPLATE = IOC_ROOT + "HDF:FileTemplate"
HDF_PATH = IOC_ROOT + "HDF:FilePath"
HDF_NAME = IOC_ROOT + "HDF:FileName"
HDF_CAPTURE = IOC_ROOT + "HDF:Capture"

HDF_MODE = IOC_ROOT + "HDF:FileWriteMode"
STREAM_MODE = 2

# Read Run Number #############################################################

if reset_file_num:
    caput(HDF_FILE_NUM.format(ALL_FEMS), 1)
if len(set(caget([HDF_NUM.format(fem) for fem in FEMS]))) != 1:
    raise IOError("NumCapture does not match on each node of detector")
run_number = caget(HDF_FILE_NUM.format(1))

# Create Virtual Dataset ######################################################

print("Creating VDS...")
file_prefix = file_name + "{:05d}_".format(run_number)
files = ["{prefix}{fem}.hdf5".format(prefix=file_prefix, fem=fem)
         for fem in FEMS]

PYTHON_ANACONDA = "/dls_sw/apps/python/anaconda/1.7.0/64/bin/python"
VDS_GEN = "/home/mef65357/Detectors/VDS/vds-gen/vdsgen/vdsgen.py"
EMPTY = "-e"
FILES = "-f"
FRAMES = "--frames"
STRIPE_SPACING = "-s"
MODULE_SPACING = "-m"
DATA_PATH = "-d"
HEIGHT = "--height"
WIDTH = "--width"
DATA_TYPE = "--data_type"

# Base arguments
command = [PYTHON_ANACONDA, VDS_GEN, file_path]
# Define empty and required arguments to do so
command += [EMPTY,
            FILES] + files + \
           [FRAMES, str(num_frames),
            HEIGHT, str(stripe_height),
            WIDTH, str(stripe_width),
            DATA_TYPE, data_type]
# Override default spacing and data path
command += [STRIPE_SPACING, "3",
            MODULE_SPACING, "127",
            DATA_PATH, "entry/data/data"]

subprocess.call(command)

# from vdsgen import generate_vds
#
# source_metadata = dict(frames=num_frames, height=256, width=1024,
#                        data_type="int16")
# generate_vds(file_path, files=files, data_path="entry/data/data",
#              stripe_spacing=3, module_spacing=127, source=source_metadata)

# Trigger Acquisition #########################################################

print("Setting parameters")

caput(HDF_NUM.format(ALL_FEMS), num_frames)
caput(ACQ_NUM.format(ALL_FEMS), num_frames)
if num_frames > 1:
    caput(ACQ_MODE.format(ALL_FEMS), MULTIPLE_MODE)
else:
    caput(ACQ_MODE.format(ALL_FEMS), SINGLE_MODE)

caput(HDF_ENABLE.format(ALL_FEMS), 1)
caput(HDF_PATH.format(ALL_FEMS), file_path, datatype=DBR_CHAR_STR)
caput(HDF_NAME.format(ALL_FEMS), file_name, datatype=DBR_CHAR_STR)
caput(HDF_MODE.format(ALL_FEMS), STREAM_MODE)
caput(ACQ_TIME.format(ALL_FEMS), acquire_time)

caput([HDF_INPUT.format(fem) for fem in FEMS],
      ["det{}.fem".format(fem) for fem in FEMS])
caput([HDF_TEMPLATE.format(fem) for fem in FEMS],
      ["%s%s%05d_{}.hdf5".format(fem) for fem in FEMS], datatype=DBR_CHAR_STR)

print("Starting acquisition")
caput(ACQUIRE.format(ALL_FEMS), 1)
caput(HDF_CAPTURE.format(ALL_FEMS), 1)

print("Waiting for acquisition...")
while True:
    sleep(1)
    if 0 in caget([HDF_CAPTURE.format(fem) for fem in FEMS]):
        break

print("Acquisition complete.")

# Create Virtual Dataset ######################################################

# print("Creating VDS...")
# file_prefix = file_name + "{:05d}_".format(run_number)
#
# PYTHON_ANACONDA = "/dls_sw/apps/python/anaconda/1.7.0/64/bin/python"
# VDS_GEN = "/home/mef65357/Detectors/VDS/vds-gen/vdsgen/vdsgen.py"
# subprocess.call([PYTHON_ANACONDA, VDS_GEN, file_path, file_prefix])

# # from vdsgen import generate_vds
# # generate_vds(file_path, file_prefix)
