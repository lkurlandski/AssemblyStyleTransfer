"""
Globals
"""


BR = "|" + "=" * 98 + "|"

BUCKET = "s3://sorel-20m/09-DEC-2020/binaries/"
WINDOWS_BUCKET = "'lk3591@armitage.csec.rit.edu:/home/lk3591/Documents/datasets/Windows/extracted/*'"
UPX = "upx"
AWS = "aws"

SPECIALS = [
    UNK := chr(0x00c0),  # À - unknown
    MSK := chr(0x00c1),  # Á - mask
    PAD := chr(0x00c2),  # Â - padding
    SEP := chr(0x00c3),  # Ã - separator
    CLS := chr(0x00c4),  # Ä - classification
    BOS := chr(0x00c5),  # Å - beginning of sequence
    EOS := chr(0x00c6),  # Æ - end of sequence
]
NONSPECIALS = [
    ADR := chr(0x00c7),  # Ç - address
    STR := chr(0x00c8),  # È - begins with str.
    VAR := chr(0x00c9),  # É - begins with var.
    SYM := chr(0x00ca),  # Ê - begins with sym.
    FCN := chr(0x00cb),  # Ë - begins with fcn.
    ARG := chr(0x00cc),  # Ì - begins with arg_ or ARG_
    SUB := chr(0x00cd),  # Í - begins with sub.
    ASM := chr(0x00ce),  # Î - begins with asm.
    VTABLE := chr(0x00cf),  # Ï - begins with vtable.
    SWITCH := chr(0x00d0),  # Ð - begins with switch.
    CASE := chr(0x00d1),  # Ñ - begins with case.
    NUM := chr(0x00d2),  # Ò - a number
    INT := chr(0x00d3), # Ó - begins with int.
    INVALID := chr(0x00d4), # Ô - invalid
    RELOC := chr(0x00d5), # Õ - begins with reloc.
    SECTION := chr(0x00d6), # Ö - begins with section.
    STD := chr(0x00d7), # × - begins with std::
    LOC := chr(0x00d8), # Ø - begins with loc.
    INS := "\n", # instruction separator
]
