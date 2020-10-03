import pyopencl
import os
import numpy

"""
func parseplatforms:
    returns list of available OpenCL platforms

    Arguments:
    none

    Process:
    return a list of all platfroms found by pyopencl
"""
def listplatforms():
    ocl_platforms=[platform.name for platform in pyopencl.get_platforms()]
    return ocl_platforms
