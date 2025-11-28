from ndsolver import core, hdf5
from tests import test_domains
import cProfile
import pstats
import time

def converge_150():
    profiler = cProfile.Profile()

    start_time = time.time()
    profiler.enable()
    s = test_domains.test_solid

    test_sol = core.Solver(s, (1,0), sol_method="bicgstab")
    test_sol.converge()
    profiler.disable()

    profiler.dump_stats("150-test-profile.prof")
    return time.time() - start_time

def converge_150_direct():
    profiler = cProfile.Profile()

    start_time = time.time()
    profiler.enable()
    s = test_domains.test_solid

    test_sol = core.Solver(s, (1,0))
    test_sol.monolithic_solve()
    profiler.disable()

    profiler.dump_stats("150-test-profile.prof")
    return time.time() - start_time


def print_profile(profile_path):
    stats = pstats.Stats(profile_path)
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    stats.print_stats(40)

if __name__ == "__main__":
    print("Running a stress test . . .")
    fin_time = converge_150()
    print(f"Time to finish: {fin_time}")
    print("Profile written to '150-test-profile.prof'")
