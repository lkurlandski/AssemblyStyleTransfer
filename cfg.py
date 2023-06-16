"""
Globals
"""

import capstone
import torch


BUCKET = "s3://sorel-20m/09-DEC-2020/binaries/"
UPX = "/home/lk3591/.local/share/upx-4.0.2-amd64_linux/upx"  # TODO: require user to have upx on the $PATH
AWS = "/home/lk3591/anaconda3/envs/AssemblyStyleTransfer/bin/aws"
ARCH = capstone.CS_ARCH_X86
MODE = capstone.CS_MODE_32

UNK = "<UNK>"
MSK = "<MSK>"
PAD = "<PAD>"
SEP = "<SEP>"
CLS = "<CLS>"
BOS = "<BOS>"
EOS = "<EOS>"
ADR = "<ADR>"
STR = "<STR>"
SPECIALS = [UNK, MSK, PAD, SEP, CLS, BOS, EOS, ADR, STR]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
