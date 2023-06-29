"""
Globals
"""

import os

import capstone
import torch


N_WORKERS = len(os.sched_getaffinity(0)) - 1


BUCKET = "s3://sorel-20m/09-DEC-2020/binaries/"
WINDOWS_BUCKET = "'lk3591@armitage.csec.rit.edu:/home/lk3591/Documents/datasets/Windows/extracted/*'"
UPX = "upx"
AWS = "aws"
ARCH = capstone.CS_ARCH_X86
MODE = capstone.CS_MODE_32

UNK = "<UNK>"
MSK = "<MSK>"
PAD = "<PAD>"
SEP = "<SEP>"
CLS = "<CLS>"
BOS = "<BOS>"
EOS = "<EOS>"
SPECIALS = [UNK, MSK, PAD, SEP, CLS, BOS, EOS]

ADR = "<ADR>"
STR = "<STR>"
VAR = "<VAR>"
SYM = "<SYM>"
FCN = "<FCN>"
ARG = "<ARG>"
SUB = "<SUB>"
ASM = "<ASM>"
VTABLE = "<VTABLE>"
SWITCH = "<SWITCH>"
CASE = "<CASE>"
NUM = "<NUM>"
NONSPECIALS = [ADR, STR, VAR, SYM, FCN, ARG, SUB, ASM, VTABLE, CASE, NUM]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_BASE_A100 = 64
