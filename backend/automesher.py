#--------- READ ME----------------------------------------------------------------
# A package containing automesher functions.
#---------------------------------------------------------------------------------

import numpy as np
import geometrics as gmt
import functbook as fnb
import domain as dm


# Helper function
def getrings(md: dm.MeshDomain, shape_i: int) -> list:
    """
    Get curve rings of all the sequences that make a shape.

    Args:
        md: MeshDomain
        shape_i: shape index

    Returns:
        list containing all the curve lists of the sequence indexes
    
    """
    sqlist = list(md.shapes[shape_i])
    # while sqlist not empty
    # next sequence = first sequence
    # while next sequence not in ringlist
    # add it, get next sequence
    # after break: add ringlist to seqlist
    # return


def automesher():
    """
    The thing that does the thing.
    """













