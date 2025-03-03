#--------- READ ME----------------------------------------------------------------
# A package containing automesher functions.
#---------------------------------------------------------------------------------

import numpy as np
import geometrics as gmt
import functbook as fnb
import domain as dm
import gmsh
import sys
from typing import Callable


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


def rework_geo(md: dm.MeshDomain, tol = 10**-2) -> dm.MeshDomain:
    """
    Rework a mesh domain so that curves are represented by bspline control points, and the geometry can be used in the md2gmsh_2.
    
    Args:
        md: MeshDomain

    Returns:
        reworked MeshDomain

    """
    points = md.points
    squencs = md.squencs
    spacing = np.array(md.spacing)

    newpoints = np.array([], dtype=float).reshape(0,2)
    newsquencs = []
    newspacing = []
    remi = []

    # Get all sequences and find their controlpts
    for i in range(len(squencs)):
        squence = squencs[i]
        if len(squence) <= 3:
            newseq = list(range(len(newpoints), len(newpoints) + len(squence)))
            newpoints = np.vstack((newpoints, points[squence]))
            newsquencs.append(newseq)
            newspacing = np.hstack((newspacing, spacing[squence]))
            remi = remi + squence
        else:
            newpts = gmt.fitctrlpts(points[squence])
            newseq = list(range(len(newpoints), len(newpoints) + len(newpts)))
            newpoints = np.vstack((newpoints, newpts))
            newsquencs.append(newseq)
            newspacing = np.hstack((newspacing, np.full(len(newpts), None)))

    # create new mesh domain
    newmd = dm.MeshDomain(newpoints, newsquencs, md.shapes, list(newspacing), md.nodes, md.mesh_types)
    # clear duplicates
    newmd.clear_duplicates(tol)
    # get all points with non-None spacing, that werent used, and add them
    nonei = np.nonzero(spacing == None)[0]
    deletei = np.unique(np.concatenate((nonei, remi)))
    addpoints = np.delete(points, deletei, axis=0)
    addspacing = list(np.delete(spacing, deletei))

    if len(addpoints) > 0:
        # add points and spacings
        newmd.points = np.vstack((newmd.points, addpoints))
        newmd.spacing = newmd.spacing + addspacing

    return newmd


def md2gmsh_1(md: dm.MeshDomain):
    """
    Translate a MeshDomain to gmsh model.
    """
    # add 0d
    for i in range(len(md.points)):
        x, y = md.points[i]
        spc = md.spacing[i]
        if spc == None:
            gmsh.model.geo.addPoint(x, y, 0, tag=i+1)
        else:
            gmsh.model.geo.addPoint(x, y, 0, spc, tag=i+1)

    # add 1d
    for i in range(len(md.squencs)):
        squence = list(np.array(md.squencs[i]) + 1)
        node = md.nodes[i]
        if len(squence) == 2:
            gmsh.model.geo.addLine(squence[0], squence[-1], i+1)
        elif len(squence) > 2:
            gmsh.model.geo.addSpline(squence, i+1)
        if node != None:
            gmsh.model.geo.mesh.setTransfiniteCurve(i+1, node[0], node[1], node[2])
    
    # add 2d
    cj = len(md.squencs) + 1
    for i in range(len(md.shapes)):
        mesh_type = md.mesh_types[i]
        looptags = []
        if mesh_type == 'void':
            continue
        rings, signs = getrings(md, i)
        for irs in range(len(rings)):
            loop = list((rings[irs] + 1) * signs[irs])
            gmsh.model.geo.addCurveLoop(loop, cj)
            looptags.append(cj)
            cj += 1
        gmsh.model.geo.addPlaneSurface(looptags, i+1)
        if mesh_type == 'hex':
            gmsh.model.geo.mesh.setTransfiniteSurface(i+1)
            gmsh.model.geo.mesh.setRecombine(2, i+1)
    
    gmsh.model.geo.synchronize()


def md2gmsh_2(md: dm.MeshDomain, synch: bool = True):
    """
    Translate a MeshDomain to gmsh model with BSplines.
    """
    md = rework_geo(md)
    # add 0d
    for i in range(len(md.points)):
        x, y = md.points[i]
        spc = md.spacing[i]
        if spc == None:
            gmsh.model.geo.addPoint(x, y, 0, tag=i+1)
        else:
            gmsh.model.geo.addPoint(x, y, 0, spc, tag=i+1)

    # add 1d
    for i in range(len(md.squencs)):
        squence = list(np.array(md.squencs[i]) + 1)
        node = md.nodes[i]
        if len(squence) <= 3:
            gmsh.model.geo.addLine(squence[0], squence[-1], i+1)
        elif len(squence) > 3:
            gmsh.model.geo.addBSpline(squence, i+1)
        if node != None:
            gmsh.model.geo.mesh.setTransfiniteCurve(i+1, node[0], node[1], node[2])
    
    # add 2d
    cj = len(md.squencs) + 1
    for i in range(len(md.shapes)):
        mesh_type = md.mesh_types[i]
        looptags = []
        if mesh_type == 'void':
            continue
        rings, signs = getrings(md, i)
        for irs in range(len(rings)):
            loop = list((rings[irs] + 1) * signs[irs])
            gmsh.model.geo.addCurveLoop(loop, cj)
            looptags.append(cj)
            cj += 1
        gmsh.model.geo.addPlaneSurface(looptags, i+1)
        if mesh_type == 'hex':
            gmsh.model.geo.mesh.setTransfiniteSurface(i+1)
            gmsh.model.geo.mesh.setRecombine(2, i+1)
    
    gmsh.model.geo.synchronize()


def md2gmsh_3(md: dm.MeshDomain):
    """
    Translate a MeshDomain to gmsh model using OpenCASCADE kernel.
    """
    # add 0d
    for i in range(len(md.points)):
        x, y = md.points[i]
        spc = md.spacing[i]
        if spc == None:
            gmsh.model.occ.addPoint(x, y, 0, tag=i+1)
        else:
            gmsh.model.occ.addPoint(x, y, 0, spc, tag=i+1)

    # add 1d
    for i in range(len(md.squencs)):
        squence = list(np.array(md.squencs[i]) + 1)
        if len(squence) == 2:
            gmsh.model.occ.addLine(squence[0], squence[-1], i+1)
        elif len(squence) > 2:
            gmsh.model.occ.addSpline(squence, i+1)

    # add 2d
    cj = len(md.squencs) + 1
    for i in range(len(md.shapes)):
        looptags = []
        mesh_type = md.mesh_types[i]
        if mesh_type == 'void':
            continue
        rings, signs = getrings(md, i)
        for irs in range(len(rings)):
            loop = list((rings[irs] + 1))
            gmsh.model.occ.addCurveLoop(loop, cj)
            looptags.append(cj)
            cj += 1
        gmsh.model.occ.addPlaneSurface(looptags, i+1)
        cvtag = i+1
    
    gmsh.model.occ.synchronize()

    # set transfinite
    for i in range(len(md.squencs)):
        node = md.nodes[i]
        if node != None:
            gmsh.model.mesh.setTransfiniteCurve(i+1, node[0], node[1], node[2])

    for i in range(len(md.shapes)):
        mesh_type = md.mesh_types[i]
        if mesh_type == 'hex':
            gmsh.model.mesh.setTransfiniteSurface(i+1)
            gmsh.model.mesh.setRecombine(2, i+1)

    # embed stray points
    spi = np.array(range(len(md.points)))
    bsi = np.unique(np.concatenate(md.squencs))
    spi = list(np.array(np.delete(spi, bsi) + 1, dtype=int))
    gmsh.model.mesh.embed(0, spi, 2, cvtag)
    

def mesh(md: dm.MeshDomain, name: str, transfunc: int, savecad: bool = True, savemesh: bool = True, show: bool = True, init: bool = True, fin: bool = True):
    """
    Turn a MeshDomain into a gmsh mesh.

    Args:
        md: MeshDomain
        name: name of model
        transfunc: which function to use to translate the mesh domain to a gmsh model
        savecad: if True save the cad
        savemesh: if True save the mesh
        show: if True, show the mesh
        init: if True, initialize gmsh
        fin: if True, finalize gmsh

    """
    if init:
        gmsh.initialize()

    gmsh.model.add(name)
    # generate mesh
    if transfunc == 1:
        md2gmsh_1(md)
    elif transfunc == 2:
        md2gmsh_2(md)
    elif transfunc == 3:
        md2gmsh_3(md)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 2)
    gmsh.model.mesh.generate()
    if show:
        gmsh.fltk.run()
    if fin:
        gmsh.finalize()


