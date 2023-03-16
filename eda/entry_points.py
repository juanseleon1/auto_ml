from argparse import ArgumentParser, Namespace

from eda.eda import EDAMaker


def do_eda():
    """
    Generate an exploratory data analysis (EDA) report for a given dataset.

    Args:
        None.

    Returns:
        None.
        
    Raises:
        None.
    """
    parser = ArgumentParser(
        prog="generate_eda", description="This program does an EDA report on a Given dataset"
    )
    parser.add_argument("datafile", type=str, help="Input data file name (mandatory)")
    args: Namespace = parser.parse_args()
    eda_maker = EDAMaker()
    eda_maker.generate_eda_report(args.datafile)
