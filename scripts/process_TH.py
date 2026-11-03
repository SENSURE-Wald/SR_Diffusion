from scripts.process_LS_dop import LowerSaxonyProcessor


class ThuringiaProcessor(LowerSaxonyProcessor):
    """
    The Thuringia data comes in the form of either 5x5, 4x5 or even 5x4 km tiles.
    So we will need to compute these sizes based on the image size. As always a
    1x1 km tile has 5000x5000 px. So if we have a (20000, 25000) image this will
    end up as a (4, 5) tuple for the square_km return.

    """


if __name__ == "__main__":
    processor = ThuringiaProcessor()
    processor.process()
