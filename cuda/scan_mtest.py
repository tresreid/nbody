#! /usr/bin/env python

import shutil
import subprocess 

nmat = 1

values = []

# return value in us
def convert(val):
    """ Unit conversion hack """
    lookup = { 'ns' : 1e-3, 'us' : 1, 'ms' : 1000, 's' : 1000000 }
    unit = val[-2:]
    try:
        number = float(val[:-2])
    except ValueError:
        print 'exception!', val[:-2]
        return val
    #print unit
    if unit in lookup:
        return lookup[unit]*number
    return int(val)
        




while ( nmat < 1e8 ) :
    #print "nmat = ", nmat
    ret = subprocess.check_output(["nvprof", "./mtest", str(nmat)], stderr=subprocess.STDOUT)
    ret = ret.splitlines()
    matches = filter(lambda x:'smallMatrix' in x, ret)
    #print "matches >> ", matches
    #print len(ret)
    elem  = matches[0].split()
    #values.append([float(elem[4]),float(elem[5])])
    tavg  = convert(elem[3])
    tmin = convert(elem[4])
    matches = filter(lambda x:'DtoH' in x, ret)
    htod = convert(matches[0].split()[3])
    matches = filter(lambda x:'HtoD' in x, ret)
    dtoh = convert(matches[0].split()[3])
    matches = filter(lambda x:'NBLOCKS' in x, ret)
    nblocks = int(matches[0].split()[2])
    nthreads = int(matches[0].split()[4])
    tper = 1.0*tavg/nmat
    ## CPU TIME
    matches = filter(lambda x:'CPU delta' in x, ret)
    #print nmat,tavg,tmin,htod,dtoh,nblocks,nthreads
    cpu_t = float(matches[0].split()[4])
    cpu_t_per = float(matches[0].split()[4])/nmat

    values.append([nmat,tavg,tmin,htod,dtoh, tper,nblocks,nthreads,cpu_t,cpu_t_per])
    convert(elem[3])
    # shutil.copyfile('final.dat','final_%s_%s.dat' % (nthreads, nblocks))
    # try:
    #     shutil.move('cuda_profile_0.log', 'profile_%s_%s.log' % ( nthreads, nblocks))
    # except IOError as err:
    #     print 'copy failed: ', err
    nmat = nmat * 10


# print '\n\n\n\n'
# print values
# print '\n\n\n\n'

for v in values:
    print "% 9d %8.2g %8.2g %8.2g %8.2g %8.2g %6d % 5d %8.2g %8.2g" % tuple(v)
    # for fv in v:
    #     print fv, ' ',
    # print


