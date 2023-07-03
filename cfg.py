"""
Globals
"""

import os

import capstone
import torch


N_WORKERS = max(len(os.sched_getaffinity(0)) - 1, 1)


BUCKET = "s3://sorel-20m/09-DEC-2020/binaries/"
WINDOWS_BUCKET = "'lk3591@armitage.csec.rit.edu:/home/lk3591/Documents/datasets/Windows/extracted/*'"
UPX = "upx"
AWS = "aws"
ARCH = capstone.CS_ARCH_X86
MODE = capstone.CS_MODE_32

UNK = chr(0x00c0)  # À
MSK = chr(0x00c1)  # Á
PAD = chr(0x00c2)  # Â
SEP = chr(0x00c3)  # Ã
CLS = chr(0x00c4)  # Ä
BOS = chr(0x00c5)  # Å
EOS = chr(0x00c6)  # Æ
SPECIALS = [UNK, MSK, PAD, SEP, CLS, BOS, EOS]

ADR = chr(0x00c7)  # Ç
STR = chr(0x00c8)  # È
VAR = chr(0x00c9)  # É
SYM = chr(0x00ca)  # Ê
FCN = chr(0x00cb)  # Ë
ARG = chr(0x00cc)  # Ì
SUB = chr(0x00cd)  # Í
ASM = chr(0x00ce)  # Î
VTABLE = chr(0x00cf)  # Ï
SWITCH = chr(0x00d0)  # Ð
CASE = chr(0x00d1)  # Ñ
NUM = chr(0x00d2)  # Ò
INT = chr(0x00d3)
INVALID = chr(0x00d4)
INS = "\n"
NONSPECIALS = [ADR, STR, VAR, SYM, FCN, ARG, SUB, ASM, VTABLE, CASE, NUM, INT, INS, INVALID]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_BASE_A100 = 64
