"""

"""


import argparse
import configparser

##############
#  settings  #
##############

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("--train-dir", type=str, required=True, help="directory of the train dataset (required)")
parser.add_argument("--val-dir", type=str, required=True, help="directory of the validation dataset (required)")
parser.add_argument("--test-dir", type=str, help="directory of the train dataset (optional)")

# training settings
parser.add_argument("--checkpoint", type=str, default=None, help="any checkpoint to resume from")
parser.add_argument("--num-workers", type=int, default=16, help="number of workers for loading data (default:16)")

# augmentation settings
parser.add_argument("--add-aug", )

# learning settings
parser.add_argument("--epochs", type=int, default=128, help="training epochs (default: 128)")
parser.add_argument("--batch-size", type=int, default=32, help="batch size to load data (default: 32)")
parser.add_argument("--lr-init", )
parser.add_argument("--lr-schedule", )

# evaluation settings
parser.add_argument("--loss-func", type=str, )
parser.add_argument("--add-metric", type=str, action="append")

# log settings
parser.add_argument("--slack-channel", type=str, default="#botlog", help="Slack channel that will receive log")
parser.add_argument("--debug", default=False, action="store_true", help="")



if __name__ == "__main__":
    args = parser.parse_args()
