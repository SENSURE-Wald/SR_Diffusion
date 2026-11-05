from scripts.process_LS_dop import LowerSaxonyProcessor


class BavariaProcessor(LowerSaxonyProcessor):
    """
    The output is consistent with the Lower Saxony name scheme and therefore this is just a rename.
    """

    pass


if __name__ == "__main__":
    processor = BavariaProcessor()
    processor.process()
