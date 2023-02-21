from opensfm import reconstruction
from opensfm.actions import reconstruct

from . import command
import argparse
from opensfm.dataset import DataSet


class Command(command.CommandBase):
    name = "reconstruct"
    help = "Compute the reconstruction"

    def run_impl(self, dataset: DataSet, args: argparse.Namespace) -> None:
        reconstruct.run_dataset(dataset)

    def add_arguments_impl(self, parser: argparse.ArgumentParser) -> None:
        pass
