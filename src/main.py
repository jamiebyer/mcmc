from tests.test_inversion import *
import cProfile
from pstats import Stats, SortKey


def run_inversion(data_type="synthetic_data"):
    pass


if __name__ == "__main__":
    in_path = "./results/inversion/results-.nc"
    """
    profiling command
    python -m cProfile -o profiling_stats.prof src/main.py
    snakeviz profiling_stats.prof
    """

    test_sampling_prior(rerun=True, plot=False)
