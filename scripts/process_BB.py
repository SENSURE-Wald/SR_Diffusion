from scripts.process_LS_dop import LowerSaxonyProcessor


class BrandenBurgProcessor(LowerSaxonyProcessor):
    """
    After the output of fix_bb.py the file format is consistent with the Lower Saxony name scheme and therefore this is just a rename.
    """

    pass


if __name__ == "__main__":
    processor = BrandenBurgProcessor()
    processor.process()
