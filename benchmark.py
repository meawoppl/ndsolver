import core, hdf5, text_test
import time, hotshot
# NB: hotshot.stats is imported in the print profile
# Function (see note there)

def converge_150():
    prof = hotshot.Profile("150-test-profile.prof")

    start_time = time.time()
    prof.start()
    s = text_test.test_array

    test_sol = core.Solver(s, (1,0), printing = 2, sol_method = "bicgstab")
    test_sol.converge()
    prof.stop()

    return time.time() - start_time

def converge_150_direct():
    prof = hotshot.Profile("150-test-profile.prof")

    start_time = time.time()
    prof.start()
    s = text_test.test_array

    test_sol = core.Solver(s, (1,0), printing = 2)
    test_sol.monolithic_solve()
    prof.stop()

    return time.time() - start_time


def print_profile(profile_path):
    # .stats needs extra package (python-profile)
    # so relative import is here
    # Ergo stats analysis on home computer only!
    import hotshot.stats
    stats = hotshot.stats.load(profile_path)
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    stats.print_stats(40)

if __name__ == "__main__":
    print "Running a stress test . . ."
    fin_time = converge_150()
    print "Time to finish:", fin_time
    print "Profile written to '150-test-profile.prof'"
