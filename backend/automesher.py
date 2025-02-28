#--------- READ ME----------------------------------------------------------------
# A package containing automesher functions.
#---------------------------------------------------------------------------------

import numpy as np
import geometrics as gmt
import functbook as fnb
import domain as dm
import gmsh
import sys


# Helper functions
def getrings(md: dm.MeshDomain, shape_i: int) -> list:
    """
    Get curve rings of all the sequences that make a shape.

    Args:
        md: MeshDomain
        shape_i: shape index

    Returns:
        a list containing all the curve lists of the sequence indexes, and a list containing orientation signs
    
    """
    sqlist = list(md.shapes[shape_i])
    seqringlist, seqsignlist = [], []
    while len(sqlist) > 0:
        ringlist, signlist = [], []
        nsqi = sqlist[0]
        nindx = 0
        while True:
            psqi = nsqi
            ringlist.append(nsqi)
            signlist.append(2*gmt.opst_i(gmt.opst_i(nindx)) + 1)
            sqlist.remove(psqi)
            nindx = gmt.opst_i(nindx)
            nsqi = md.point_ref(md.squencs[psqi][nindx], sqlist)
            if len(nsqi) > 0:
                nsqi = nsqi[0]
                nindx = md.squencs[nsqi].index(md.squencs[psqi][nindx])
            else:
                break
            if len(sqlist) == 0:
                break
        
        seqsignlist.append(np.array(signlist))
        seqringlist.append(np.array(ringlist))

    return [seqringlist, seqsignlist]


def md2gmsh(md: dm.MeshDomain, synch: bool = True):
    """
    Translate a MeshDomain to gmsh model.
    """
    # add 0d
    for i in range(len(md.points)):
        x, y = md.points[i]
        spc = md.spacing[i]
        if spc == None:
            gmsh.model.geo.addPoint(x, y, 0, tag=i)
        else:
            gmsh.model.geo.addPoint(x, y, 0, spc, tag=i)

    # add 1d
    for i in range(len(md.squencs)):
        squence = md.squencs[i]
        node = md.nodes[i]
        if len(squence) == 2:
            gmsh.model.geo.addLine(squence[0], squence[1], i)
        elif len(squence) > 2:
            gmsh.model.geo.addSpline(squence, i)
        
        if node != None:
            gmsh.model.geo.mesh.setTransfiniteCurve(i, node[0], node[1], node[2])
    
    # add 2d
    cj = len(md.squencs)
    for i in range(len(md.shapes)):
        mesh_type = md.mesh_types[i]
        looptags = []
        if mesh_type == 'void':
            continue
        rings, signs = getrings(md, i)
        for irs in range(len(rings)):
            loop = list(rings[irs] * signs[irs])
            gmsh.model.geo.addCurveLoop(loop, cj)
            looptags.append(cj)
            cj += 1
        gmsh.model.geo.addPlaneSurface(looptags, i)
        if mesh_type == 'hex':
            gmsh.model.geo.mesh.setTransfiniteSurface(i)
            gmsh.model.geo.mesh.setRecombine(2, i)
    
    # synch
    if synch:
        gmsh.model.geo.synchronize()


def mesh(md: dm.MeshDomain, name: str, savecad: bool = True, savemesh: bool = True, show: bool = True, fin: bool = False):
    """
    Turn a MeshDomain into a gmsh mesh. 
    """
    # initialize
    gmsh.initialize()
    gmsh.model.add(name)
    # generate mesh
    md2gmsh(md)
    gmsh.model.mesh.generate()
    if show:
        gmsh.fltk.run()
    if fin:
        gmsh.finalize()




