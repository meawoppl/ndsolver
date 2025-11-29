import hashlib
import logging
import os
import time

import numpy as np
from numpy import array, pi, sqrt, zeros

from . import hdf5

logger = logging.getLogger(__name__)

sc = {2:array([[0.0,0.0]]),

      3:array([[0.0,0.0,0.0]]) }

bcc = {2:array([[0.0,0.0]]),
       
       3:array([[0.5,0.5,0.5],
                [0.0,0.0,0.0]]) }

fcc = {2:array([[0.0,0.0],
                [0.5,0.5]]),
       
       3:array([[0.0,0.0,0.0],
                [0.5,0.5,0.0],
                [0.0,0.5,0.5],
                [0.5,0.0,0.5]]) }

def pack_cp(aspect, ndradius, feature_count):
    desired_density = pi * ndradius**2 * feature_count
    logger.info(f"Desired density: {desired_density}")
    if desired_density > (pi/4.):
        raise ValueError("Density too high")

    #TODO NotDone!


def gen_h5_hash(solid_array):
    """Make a hash for the filename based on solid shape."""
    hash_generator = hashlib.md5()
    s = list(solid_array.flatten())
    string_s = ''.join([str(i) for i in s])
    hash_generator.update(string_s.encode())
    return hash_generator.hexdigest()[::4]

def name_file(fluid_domain, irad, obs_count):
    # filehash
    filehash = gen_h5_hash(fluid_domain)
    
    # shape string
    domain_shape = fluid_domain.shape
    shape_string = 'x'.join([str(i) for i in domain_shape])
    
    # dimensionality
    obs_count_str = str(obs_count)

    # Denom, to nearest int
    obs_rad_frac_denom = int(irad)

    return f"{shape_string}-{obs_rad_frac_denom}th-{obs_count_str}-{filehash}.h5"

def random_pack_nd_points(aspect, nd_radius, feature_count):
    '''Returns a (feature_count)x(len(aspect)) array of random points that are at least nd_radius apart in periodic space'''
    # Stupid Check
    for element in aspect:
        if element > 1.:
            raise ValueError("Not really ND . . .")

    ndim = len(aspect)
    nd_volume = array(aspect).prod()

    # Dont wanna symbolic this . . .
    if   ndim == 2:
        solid = feature_count * pi * nd_radius ** 2
    elif ndim == 3:
        solid = feature_count * (4./3) * pi * nd_radius**3
    else:
        solid = -0
    
    logger.info(f"Fraction: {solid/nd_volume}")

    # Smack one in the center . . .
    point_list = (array(aspect)/2.).reshape((1,ndim)).copy()
    placed_points = 1

    proposed_point = zeros(ndim)
    attempt = 0
    while placed_points < feature_count:
        # print placed_points, attempt
        # Generate a random point
        for n in range(ndim):
            proposed_point[n] = np.random.uniform(0, aspect[n])

        # Non periodic vector
        dist_vector = point_list - proposed_point

        # If a distance in a given vector direction is greater than half the domain width
        # Then the real distance is the domain length minus the apperent distance
        for dim in range(ndim):
            to_subtract = abs(dist_vector[:,dim]) > (aspect[dim]/2.)
            dist_vector[to_subtract,dim] = aspect[dim] - abs(dist_vector[to_subtract,dim])

        # Find Vector Magnitude 
        dist_vector = dist_vector ** 2
        dist = sqrt(dist_vector.sum(axis=1))
        
        too_close = dist < (2 * nd_radius)
        if any(too_close):
            attempt += 1
            continue
        else:
            point_list = np.r_[point_list, proposed_point.reshape((1,ndim)).copy()]
            placed_points += 1
            # print placed_points, "placed points", attempt, "previous attempts in this step."
            attempt = 0

    return point_list

def ndpattern(points, shape):
    '''Pattern nd-points'''
    points = points.copy()
    for dim, num in enumerate(shape):
        for mult in range(num):
            if mult == 0:
                continue
            temp_points = points.copy()        
            temp_points[:,dim] += 1
            points = np.concatenate((points, temp_points), axis=0)
    return points


def nd_points_to_domain(points, nd_radius, domain_shape):
    domain = zeros(domain_shape, dtype=np.int8)
    ndim = len(domain_shape)
    mgrid_slices = tuple([slice(0,1,1j*n) for n in domain_shape])
    location_grids = np.mgrid[mgrid_slices]
    point_semi_shape = tuple( [ndim] + [1]*ndim )
    for point in points:
        point = point.reshape(point_semi_shape)
        dists = location_grids - point
        
        for dim in range(ndim):
            to_sub = abs(dists[dim,:]) >= 0.5
            dists[dim,to_sub] = 1 - abs(dists[dim,to_sub])
        dists **= 2
        mag = sqrt(dists.sum(axis=0))
        domain[mag <= nd_radius] = 1

    return domain

def geom_to_h5(domain_shape, feature_count, nd_radius, out_filename=None, prefix=None):
    # Calculate the aspect of the domain (usually 1,1[,1])
    aspect = tuple(array(domain_shape)/domain_shape[0])
    
    # Generate the random points
    logger.info("Packing...")
    t = time.time()
    points = random_pack_nd_points(aspect, nd_radius, feature_count)
    logger.debug(f"{time.time() - t} seconds")
    
    # Generate the actual solid array
    logger.info("Creating array...")
    t = time.time()
    solid_array = nd_points_to_domain(points, nd_radius, domain_shape)
    logger.debug(f"{time.time() - t} seconds")

    # Generate the automatic name
    if out_filename is None:
        logger.info("Creating file name hash...")
        t = time.time()
        out_filename = os.path.join(prefix, name_file(solid_array, int( (1. / nd_radius) + 0.5), feature_count))
        logger.debug(f"{time.time() - t} seconds")

    # Write the solid array to the h5 file
    logger.info("Writing to h5...")
    t = time.time()
    hdf5.write_S(out_filename, solid_array)

    # Write the geometry
    hdf5.write_geometry(out_filename, points, nd_radius * np.ones(feature_count))
    logger.debug(f"{time.time() - t} seconds")


def seimran_geom_to_h5(domain_shape, feature_count, nd_radius, out_filename):
    # Calculate the aspect of the domain (usually 1,1[,1])
    aspect = tuple(array(domain_shape)/domain_shape[0])
    
    # Generate the random points
    points = semirandom_pack_nd_points(aspect, nd_radius, feature_count)

    # Generate the actual solid array
    solid_array = nd_points_to_domain(points, nd_radius, domain_shape)

    # Write the solid array to the h5 file
    hdf5.write_S(out_filename, solid_array)

    # Write the geometry
    hdf5.write_geometry(out_filename, points, nd_radius * np.ones(feature_count))

    

def hypersphere(filename):
    s = zeros((25,25,25,25))
    x, y, z, q = np.mgrid[-1:1:25j,-1:1:25j,-1:1:25j,-1:1:25j]
    dist = sqrt(x**2 + y**2 + z**2 + q**2)
    s[dist < dist] = 1.

    hdf5.write_S(filename, s)

def points_to_sim(filename, points, ndradius, extent):
    point_count = points.shape[0]
    domain = nd_points_to_domain(points, ndradius, extent)
    logger.info(f"Domain Fraction: {domain.mean()}")
    hdf5.write_S(filename, domain)
    hdf5.write_geometry(filename, points, ndradius * np.ones(point_count))


def make_3d_sc_series(folder_to_save, number, resolution):
    radaii = np.linspace(0, 0.5, number+1)[0:]

    for num, radius in enumerate(radaii):
        logger.debug(f"Processing {num}")
        domain = nd_points_to_domain(sc[3], radius, resolution)    
        filename = os.path.join(folder_to_save, str(num) + ".h5")

        logger.info(f"Writing for fraction: {domain.mean()}")

        hdf5.write_S(filename, domain)
        hdf5.write_geometry(filename, fcc[3], radius * np.ones(4))


def make_big_3d_series():
    '''Make a domain of size (tuple) with various ndradaii, and scaled numbers'''
    size = (150,150,150)
    for n in np.linspace(10,80,8):
        for r, mult in zip([10,20,40], [1, 8, 64]):
            for x in range(20):
                logger.debug(f"n={n}, r={r}, x={x}")
                geom_to_h5(size, n * mult, 1. / r, prefix="/media/raid/fluids-h5/small-new-3d-series")
            


def make_3d_fcc_series(folder_to_save, number, resolution):
    radaii = np.linspace(0, 0.25 * sqrt(2), number+1)[0:]

    for num, radius in enumerate(radaii):
        logger.debug(f"Processing {num}")
        domain = nd_points_to_domain(fcc[3], radius, resolution)    
        filename = os.path.join(folder_to_save, str(num) + ".h5")

        logger.info(f"Writing for fraction: {domain.mean()}")

        hdf5.write_S(filename, domain)
        hdf5.write_geometry(filename, fcc[3], radius * np.ones(4))

def make_3d_bcc_series(folder_to_save, number, resolution):
    radaii = np.linspace(0, 0.25 * sqrt(3), number+1)[0:]

    for num, radius in enumerate(radaii):
        logger.debug(f"Processing {num}")
        domain = nd_points_to_domain(bcc[3], radius, resolution)    
        filename = os.path.join(folder_to_save, str(num) + ".h5")

        logger.info(f"Writing for fraction: {domain.mean()}")

        hdf5.write_S(filename, domain)
        hdf5.write_geometry(filename, fcc[3], radius * np.ones(4))

def semirandom_pack_nd_points(aspect, nd_radius, feature_count, flakeat=250000):
    '''Returns a (feature_count)x(len(aspect)) array of random points that are at least nd_radius apart in periodic space'''
    # Stupid Check
    for element in aspect:
        if element > 1.:
            raise ValueError("Not really ND . . .")

    ndim = len(aspect)
    nd_volume = array(aspect).prod()

    # Dont wanna symbolic this . . .
    if   ndim == 2:
        solid = feature_count * pi * nd_radius ** 2
    elif ndim == 3:
        solid = feature_count * (4./3) * pi * nd_radius**3
    else:
        solid = -0
    
    logger.info(f"Fraction: {solid/nd_volume}")

    # Smack one in the center . . .
    point_list = (array(aspect)/2.).reshape((1,ndim)).copy()
    placed_points = 1

    proposed_point = zeros(ndim)
    attempt = 0
    while placed_points < feature_count:
        # print placed_points, attempt
        # Generate a random point
        for n in range(ndim):
            proposed_point[n] = np.random.uniform(0, aspect[n])

        # Non periodic vector
        dist_vector = point_list - proposed_point

        # If a distance in a given vector direction is greater than half the domain width
        # Then the real distance is the domain length minus the apperent distance
        for dim in range(ndim):
            to_subtract = abs(dist_vector[:,dim]) > (aspect[dim]/2.)
            dist_vector[to_subtract,dim] = aspect[dim] - abs(dist_vector[to_subtract,dim])

        # Find Vector Magnitude 
        dist_vector = dist_vector ** 2
        dist = sqrt(dist_vector.sum(axis=1))
        
        too_close = dist < (2 * nd_radius)
        if any(too_close):
            attempt += 1
        else:
            point_list = np.r_[point_list, proposed_point.reshape((1,ndim)).copy()]
            placed_points += 1
            logger.debug(f"{placed_points} placed points, {attempt} previous attempts")
            attempt = 0

        if attempt > flakeat:
            logger.debug("Nuking a Point")
            to_nuke = np.random.randint(point_list.shape[0])
            point_list = np.r_[point_list[to_nuke::,:], point_list[:to_nuke+1:,:]]
            placed_points -= 1
            attempt = 0

    return point_list

def make_2_circle_series():
    nd_radius = 0.1
    aspect=(1.,1.)
    ndim = 2
    domain_shape = (150,150)
    
    center_point = array([[0.5,0.5]])
    for nx, dx in enumerate(np.linspace(0,0.5,100)):
        for ny, dy in enumerate(np.linspace(0,0.5,100)):
            new_point = center_point.copy()
            new_point[0,0] += dx
            new_point[0,1] += dy

            # Non periodic vector
            dist_vector = center_point - new_point

            # If a distance in a given vector direction is greater than half the domain width
            # Then the real distance is the domain length minus the apperent distance
            for dim in range(ndim):
                to_subtract = abs(dist_vector[:,dim]) > (aspect[dim]/2.)
                dist_vector[to_subtract,dim] = aspect[dim] - abs(dist_vector[to_subtract,dim])

            # Find Vector Magnitude 
            dist_vector = dist_vector ** 2
            dist = sqrt(dist_vector.sum(axis=1))
        
            too_close = dist < (2 * nd_radius)

            if too_close:
                continue

            points = np.r_[center_point, new_point]

            
            out_filename = f"explore_two/x-{nx:03d}_y-{ny:03d}.h5"
            logger.info(f"Output: {out_filename}")
            # Generate the actual solid array
            solid_array = nd_points_to_domain(points, nd_radius, domain_shape)

            # Write the solid array to the h5 file
            hdf5.write_S(out_filename, solid_array)

            # Write the geometry
            hdf5.write_geometry(out_filename, points, nd_radius * np.ones(2))
            
def make_3d_fcc_mesh_refine_study(smallest=10, largest=100, steps=10):
    mesh_sizes = np.linspace(smallest, largest, steps)

    radius = sqrt(2) / 4.
    for mesh_size in mesh_sizes:
        # Make the solid
        resolution = (mesh_size, mesh_size, mesh_size)
        domain = nd_points_to_domain(fcc[3], radius, resolution)
        
        mesh_str = f"{resolution[0]}x{resolution[1]}x{resolution[2]}"
        logger.info(f"Writing for fraction: {domain.mean()} ({mesh_str})")

        filename = os.path.join("fcc-meshrefine", mesh_str + ".h5")
        hdf5.write_S(filename, domain)
        hdf5.write_geometry(filename, fcc[3], radius * np.ones(4))


def make_dilute_meshstudy(start=10, stop=60, steps=11):
    for x in np.linspace(start,stop,steps):
        logger.info(f"Writing {x}")
        domain = zeros((x,x,x))
        domain[0,0,0] = 1

        mesh_str = f"{int(x)}x{int(x)}x{int(x)}.h5"
        filename = os.path.join("dilute-refine", mesh_str)        

        hdf5.write_S(filename, domain)


def make_2d_cyl_validation_set(folder_to_save, lower=0.01, upper = 0.49, steps=100, shape = (200,200)):
    radii = np.linspace(lower, upper, steps)

    for num, radius in enumerate(radii):
        filename = os.path.join(folder_to_save, str(num) + ".h5")
        logger.debug(f"Processing {num}: {filename}")
        domain = nd_points_to_domain(array([[0.5,0.5]]), radius, shape)    

        logger.info(f"Writing for fraction: {domain.mean()}")

        hdf5.write_S(filename, domain)
        hdf5.write_geometry(filename, fcc[3], radius * np.ones(4))

def make_3d_bcc_mesh_refine_study(smallest=10, largest=80, steps=8):
    mesh_sizes = np.linspace(smallest, largest, steps)

    logger.debug(f"Mesh sizes: {mesh_sizes}")

    # Match exactly zick's concentration
    conc = 0.6
    radius = (conc * (3./(8 * pi)))**(1./3)

    for mesh_size in mesh_sizes:
        # Make the solid
        resolution = (mesh_size, mesh_size, mesh_size)
        domain = nd_points_to_domain(bcc[3], radius, resolution)
        
        mesh_str = f"{resolution[0]}x{resolution[1]}x{resolution[2]}"
        logger.info(f"Writing for fraction: {domain.mean()} ({mesh_str})")

        filename = os.path.join("bcc-final-0.6-final-refine", mesh_str + ".h5")
        hdf5.write_S(filename, domain)
        hdf5.write_geometry(filename, bcc[3], radius * np.ones(2))

def make_halfpipe_meshrefine(smallest=10, largest=100, steps=10):
    mesh_sizes = np.linspace(smallest, largest, steps)

    logger.debug(f"Mesh sizes: {mesh_sizes}")

    # Match exactly zick's concentration
    for mesh_size in mesh_sizes:
        # Make the solid
        resolution = (mesh_size, mesh_size, mesh_size)
        x, y, z = np.mgrid[-1:1:1j*resolution[0],
                        -1:1:1j*resolution[0],
                        -1:1:1j*resolution[0]]

        domain = 1.0 * ((x**2 + y**2) > 0.5**2)

        mesh_str = f"{resolution[0]}x{resolution[1]}x{resolution[2]}"
        logger.info(f"Writing for fraction: {domain.mean()} ({mesh_str})")

        filename = os.path.join("halfpipe", mesh_str + ".h5")
        hdf5.write_S(filename, domain)


if __name__ == "__main__":
    make_big_3d_series()
    pass
    # make_2d_cyl_validation_set("2d-cylinder-validation-set")
    # make_dilute_meshstudy()
    # hypersphere("4d-hypersphere.h5")
    # make_3d_bcc_mesh_refine_study()
    # make_2_circle_series()

    # make_3d_sc_series("sc-valid-75",   25, (75,75,75))
    # make_3d_fcc_series("fcc-valid-75", 25, (75,75,75))
    # make_3d_bcc_series("bcc-valid-75", 25, (75,75,75))
    # 
    # for x in range(10):
    #     seimran_geom_to_h5((50,50,50), 720, 0.05, "semirandom/20th-%04i-%04i.h5"  % (720, x) )
    #     seimran_geom_to_h5((50,50,50), 740, 0.05, "semirandom/20th-%04i-%04i.h5"  % (740, x) )
        
    # count = list(arange(1,16) * 10)
    # for num in count:
    #     for x in range(100):
    #         print "Packing", num, x
    #         small_num = num * 8
    #         geom_to_h5((50,50,50), small_num, 0.05, "/media/raid/fluids-h5/3d/big-3d-set/20th-%04i-%04i.h5"  % (small_num, x) )
    #         geom_to_h5((50,50,50), num, 0.10,       "/media/raid/fluids-h5/3d/big-3d-set/10th-%04i-%04i.h5"  % (num, x) )
    #         print "Done!"
    
    pass
