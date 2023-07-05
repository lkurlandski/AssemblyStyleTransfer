"""
Globals
"""


BR = "|" + "=" * 98 + "|"

BUCKET = "s3://sorel-20m/09-DEC-2020/binaries/"
WINDOWS_BUCKET = "'lk3591@armitage.csec.rit.edu:/home/lk3591/Documents/datasets/Windows/extracted/*'"
UPX = "upx"
AWS = "aws"

SPECIALS = [
    UNK := chr(0x00c0),  # À
    MSK := chr(0x00c1),  # Á
    PAD := chr(0x00c2),  # Â
    SEP := chr(0x00c3),  # Ã
    CLS := chr(0x00c4),  # Ä
    BOS := chr(0x00c5),  # Å
    EOS := chr(0x00c6),  # Æ
]
NONSPECIALS = [
    ADR := chr(0x00c7),  # Ç
    STR := chr(0x00c8),  # È
    VAR := chr(0x00c9),  # É
    SYM := chr(0x00ca),  # Ê
    FCN := chr(0x00cb),  # Ë
    ARG := chr(0x00cc),  # Ì
    SUB := chr(0x00cd),  # Í
    ASM := chr(0x00ce),  # Î
    VTABLE := chr(0x00cf),  # Ï
    SWITCH := chr(0x00d0),  # Ð
    CASE := chr(0x00d1),  # Ñ
    NUM := chr(0x00d2),  # Ò
    INT := chr(0x00d3),
    INVALID := chr(0x00d4),
    INS := "\n",
]
