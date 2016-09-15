#!/bin/env dls-python

import sys
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    return parser.parse_args()


def main():

    args = parse_args()


if __name__ == "__main__":
    sys.exit(main())
