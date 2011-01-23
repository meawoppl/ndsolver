from scipy import *
from scipy import random, zeros
from IPython.kernel import client
from pprint import pprint
import time, os
from tables import openFile

# My libraries
from hdf5 import ipython_task_to_h5, forceGroup

remote_solve_text = open("remote-solve.py").read() 


def make_nd_task(S, dP, max_div, u = None, v = None, task_path = ""):
    # Load the solve routine and the newest ndsolver
    to_push = {"S" :S, "dP":dP, "max_div":max_div, "task_path":task_path}

    to_pull = ["P", "vgrids", "dP", "task_path", "meta"]

    return client.StringTask(remote_solve_text, push=to_push, pull=to_pull, clear_before=True, clear_after=True, retries = 2)

tc = client.TaskClient()

##############
# Test Block #
##############

for x in range(10):
    test = zeros((5,5,5))
    test[2,2,:] = 1
    dP = (1, 0, 0)
    max_div = 1.0e-8

    print "Making Task"
    test_task = make_nd_task(test, dP, max_div, task_path="3d-task-test.h5")

    print "Submitting" 
    tn = tc.run(test_task)
    print "Submitted"

    print tc.queue_status()
    time.sleep(4)
    print tc.queue_status()

    print "Waiting On Task"
    tr = tc.get_task_result(tn, block = True)
    
    print "Result:", tr
    if hasattr(tr.failure, "printTraceback"): 
        pprint("Teh fialz ***********")
        tr.failure.printTraceback()
        pprint("Teh fialz ***********")
        raise ValueError("wah!")
    else: 
        pprint(tr.results)
        ipython_task_to_h5(tr)

##############################
# Begin real (non-test) code #
##############################

preprocess  = []
postprocessed = set()

max_to_queue = 10

# Populate above
h5_dir = "/media/raid/fluids-h5/2d-dendrite-slices/"

# Wander through the target directory recursively, and grab all the h5 file paths
for path, folders, files in os.walk(h5_dir):
    for f in files:
        if f.endswith(".h5"):
            preprocess.append(os.path.join(path, f))

print "%i h5 files detected . . ." % len(preprocess)

# The queue is processed before loading, so that sucessful tasks are 
# not repeated when this script borks and is restarted

# NB: Still no handling of failures . . .

####################################
# Main Loop
####################################

# For each h5 we have a x-sim and a y-sim
processed_since_last_clear = 0
queued_count = len(preprocess) * 3

completed_counter = 0

while queued_count > 0:
    print time.ctime(), tc.queue_status()

    queued_count = (tc.queue_status()['scheduled'] 
                    + tc.queue_status()['pending'] 
                    + (len(preprocess) * 3) )
    print queued_count, "simulations remaining . . ."
    print completed_counter, "simulations completed . . ."

    #-----------------------------------
    # Queue Analyzing and Postprecessing
    #-----------------------------------

    # Successful tasks . . .
    tasks_to_check = set(tc.queue_status(verbose=True)['succeeded'])
    tasks_to_check.difference_update(postprocessed)

    for task_number in tasks_to_check:
        task = tc.get_task_result( task_number )        
        print "Recording sim", task.results["task_path"], "from", task.results["meta"]["Identity"]
        ipython_task_to_h5( task )
        processed_since_last_clear += 1
        postprocessed.add(task_number)

    # Failed tasks . . .
    # TODO: there is no handling of tasks that report failure with non-fatal problems
    # I can't decided if that is worth changing, basically, only flawless runs are recorded currently
    tasks_to_check = set(tc.queue_status(verbose=True)['failed'])
    
    for task_number in tasks_to_check:
        print "Failure on task #", task_number
        task = tc.get_task_result( task_number )        
        task.failure.printTraceback()
        processed_since_last_clear += 1

    #---------------
    # Queue Clearing
    #---------------

    # Clear the memory on the controller when all sucessful
    # tasks have been processed
    # It is _possible_ that between the below conditional and the
    # tc.clear() a task could slip in there.
    # restarting the system later will catch that at the expense of some small amount of computational time. . .
    qstat = tc.queue_status()
    processable = qstat['succeeded'] + qstat['failed']
    #  If there were results processed and there were not more results in the queue than processed
    if processed_since_last_clear != 0 and processable <= processed_since_last_clear:
        print "Clearing controller memory"
        print "Completed tasks:", tc.queue_status()['succeeded']
        print "Processed Tasks:", processed_since_last_clear
        tc.clear()
        processed_since_last_clear = 0

    #--------------
    # Queue Loading
    #--------------

    # If there are less than a certain number schedualed to run, and
    # we have h5's left to queue
    queue_not_full = tc.queue_status()['scheduled'] < max_to_queue
    simulations_remain = len(preprocess) > 0 
    if queue_not_full and simulations_remain:
        # Push on 2 sims, x and y for a file in preprocess
        path = preprocess.pop()
        h5_temp = openFile(path, "a")
        S = h5_temp.root.geometry.S[:]

        # If 2d or 3d
        if S.ndim == 2:
            sim_names = ["x_sim","y_sim"]
            sim_dims  = [(1,0),(0,1)]
        elif S.ndim == 3:
            sim_names = ["x_sim","y_sim","z_sim"]
            sim_dims  = [(1,0,0),(0,1,0),(0,0,1)]
        else:
            raise ValueError("WTF ^^ate (%s)" % path)

        # Check the h5 for a x-sim, y-sim (z-sim) and act the following way:
        for sim_name, dP in zip(sim_names, sim_dims):
            # This creates a sim group if it dosent already have one.
            h5_sims = forceGroup(h5_temp, "/", "simulations", 'Simulations details')
            
            # Pre-existing simulation data:
            if hasattr(h5_sims, sim_name):
                this_sim = getattr(h5_sims, sim_name)
                print "\tAlready converged. . . skipping -", path, dP
                completed_counter += 1
                continue
            # No previous solution whatsoever
            else:
                print "\tQueuing new simulation -", path, dP
                task_temp = make_nd_task(S, dP, 1e-8, task_path = path)
                tc.run( task_temp )
                
        h5_temp.close()
    elif tc.queue_status()['scheduled'] >= max_to_queue:
        print "\tAll queued up . . ."
        time.sleep(5)
        
    elif len(preprocess) == 0:
        print "\tNo h5's remaining . . ."
        time.sleep(5)
