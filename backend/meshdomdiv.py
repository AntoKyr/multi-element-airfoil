#--------- READ ME----------------------------------------------------------------
# A package containing the functions for the generation of mesh domains for a
# number of predetermined high lift arrays. 
#---------------------------------------------------------------------------------

import numpy as np
import geometrics as gmt
from typing import Union, Callable

# MESH DOMAIN CLASS
class MeshDomain(gmt.GeoShape):
    """
    Extends GeoShape, with further parametres to describe a mesh.

    Attr:
        points (ndarray): Same as superclass
        squencs (list): Same as superclass
        shapes (list): Same as superclass
        spacing (list): Contains the requested mesh element size for each point. The spacing over the entire volume is then computed by interpolating these. This is ignored in structured domains
        nodes (list): Contains an array of floats between 0 and 1, for each curve, that describes where the nodes will be placed along it, overrides spacing
        mesh_types (list): Contains strings describing the meshing method for each shape. 'hex' for structured hexa mesh, 'tri' for unstructured tri mesh, 'void' to remove domain from control volume (for example the airfoil body)

    Methods:
        YET TO BE DECLARED

    """

    def __init__(self, points: np.ndarray, squencs: list, shapes: list, spacing: list, nodes: list, mesh_types: list):
        super(MeshDomain, self).__init__(points, squencs, shapes)
        self.nodes = nodes
        self.spacing = spacing
        self.mesh_types = mesh_types


# def invert(self, sidekey):
#     """
#     Invert the sides of the domain.

#     Args:
#         sidekey (int): can be 1 or 2, pertaining to sides s11, s12 and s21, s22 respectively.

#     """
#     if sidekey == 1:
#         self.s11 = np.flipud(self.s11)
#         self.s12 = np.flipud(self.s12)
#     elif sidekey == 2:
#         self.s21 = np.flipud(self.s21)
#         self.s22 = np.flipud(self.s22)


# def flip(self):
#     """
#     Flip the s11 and s12 with the s21 and s22.
#     """
#     self.s11, self.s21 = self.s21, self.s11
#     self.s12, self.s22 = self.s22, self.s12


# SUPPORTING FUNCTIONS
def _ile(bodyc: Union[list,tuple,np.ndarray]) -> int:
    """
    Return the index of the leading edge point of the foil curve.
    """
    return np.argmax(np.hypot(bodyc[:,0] - bodyc[0,0], bodyc[:,1] - bodyc[0,1]))


def _chord(bodyc: Union[list,tuple,np.ndarray]) -> float:
    """
    Return the chord of the foil.
    """
    return np.sum((bodyc[_ile(bodyc)] - bodyc[0])**2)**0.5


def _lec(bodyc: Union[list,tuple,np.ndarray]) -> list:
    """
    Return the circle of the leading edge.
    """
    i = _ile(bodyc)
    return gmt.crcl_fit(bodyc[i-2:i+3])


# OPPOSING FACES
class FacePair:
    """
    Little helper class.

    Attr:
        face1 (list): list of indexes corresponding to a curve
        face2 (list): list of indexes corresponding to another curve

    Methods:
        flip: interchange face1 and face2    
    
    """
    def __init__(self, face1, face2):
        self.face1 = face1
        self.face2 = face2

    def flip(self):
        face1, face2 = self.face1, self.face2
        self.face1, self.face2 = face2, face1
    
    def __str__(self):
        return 'Face 1: ' + str(self.face1) + '\nFace 2: ' + str(self.face2)

from matplotlib import pyplot as plt
def opposing_faces(c1: Union[list,tuple,np.ndarray], c2: Union[list,tuple,np.ndarray], crit_func: Callable[[np.ndarray, np.ndarray, np.ndarray], bool], los_check: bool = False) -> FacePair:
    """
    Find the parts of two geometrics curves which are opposing one another by casting rays from the first curve (c1) to the second (c2). Also apply certain criteria.

    Args:
        c1: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 1
        c2: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2
        crit_func: criterion function for rays to be valid
        los_check: line of sight check, if True, checks if the ray intersects the first curve before the second, meaning its out of line of sight of the second and is discarded, not needed for simple curves
    
    Returns:
        list containing two lists, (one for each curve) which contain the indexes of the corresponding opposing faces
        
    """
    c1, c2 = np.array(c1), np.array(c2)
    # Get the closest points to figure out the correct ray casting orientation.
    dists = gmt.crv_dist(c1, c2)
    pi1 = np.argmin(np.min(dists, axis=1))
    pi2 = np.argmin(np.min(dists, axis=0))
    lfac = np.max(dists)
    # In case of curve endpoint
    if pi1 == 0:
        jp = 1
    else:
        jp = -1
    # Get curve orientation sign
    s1 = np.sign(jp) * np.sign(gmt.vectorangle(c1[pi1 + jp] - c1[pi1], c2[pi2] - c1[pi1]))
    # Generate rays
    ray1 = lfac * s1 * gmt.parallcrv(c1)
    # Cast and trace rays
    face_pair_list = []
    fbool = False
    for i in range(len(ray1)):
        ray = np.array([c1[i], ray1[i] + c1[i]])
        tris, trps = gmt.raytrace(c2, ray)
        if len(tris) > 0:
            tri, trp = tris[0], trps[0]
            # Check for self intersections
            los = True
            if los_check:
                losis = gmt.raytrace(c1, ray)[0]
                if len(losis) > 0:
                    if (losis[0] == i) or (losis[0] - 1 == i):
                        if len(losis) > 1:
                            los = False
                    else:
                        los = False
        
            if crit_func(c2[tri:tri+2], trp, ray) and los:            # apply ray criterion
                if not fbool:                      # new casting and traced face borders
                    cfb, tfb = [i, 0], [tri, 0]
                    fbool = True
                cfb[1] = i
                tfb[1] = tri
                continue

        if fbool:                                  # save face pair into list
            face1 = list(range(cfb[0], cfb[1]+1))
            if tfb[0] > tfb[1]:
                face2 = list(range(tfb[0]+1, tfb[1]-1, -1))
            else:
                face2 = list(range(tfb[0], tfb[1]+2))
            face_pair_list.append(FacePair(face1, face2))
            fbool = False

    if fbool:                                     # in case last ray traces                    
        face1 = list(range(cfb[0], cfb[1]+1))
        if tfb[0] > tfb[1]:
            face2 = list(range(tfb[0]+1, tfb[1]-1, -1))
        else:
            face2 = list(range(tfb[0], tfb[1]+2))
        face_pair_list.append(FacePair(face1, face2))

    return face_pair_list


def face_merger(face_pair_list: list, minsim_all: float = 0, minsim_1: float = 0, minsim_2: float = 0) -> list:
    """
    Merges faces from a list, according to their similarity. All similarity criteria must be passed for a merge to happen.

    Args:
        face_pair_list: the list containing the face pairs
        minsim_all: the minimum added similarity of a pair of faces with another pair, in order to be merged
        minsim_1: the minimum similarity two non-opposing faces of the first curve must have, for the two pairs to be merged
        minsim_2: the minimum similarity two non-opposing faces of the second curve must have, for the two pairs to be merged
    
    Returns:
        list of merged faces (lists of lists of indexes yada yada)

    """
    restart = True
    ll = len(face_pair_list)
    while restart:
        restart = False

        for i in range(ll):
            fpi1, fpi2 = face_pair_list[i]
            for j in range(i,ll):
                fpj1, fpj2 = face_pair_list[j]
                ls1 = len(set(fpj1 + fpi1))
                ls2 = len(set(fpj2 + fpi2))
                li1, li2 = len(fpi1), len(fpi2)
                lj1, lj2 = len(fpj1), len(fpj2)
                sim_1 = (li1 + lj1 - ls1) / min(li1, lj1)
                sim_2 = (li2 + lj2 - ls2) / min(li2, lj2)
                sim_all = (li2 + lj2 + li1 + lj1 - ls1 - ls2) / (min(li1, lj1) + min(li2, lj2))
                if (sim_1 > minsim_1) and (sim_2 > minsim_2) and (sim_all > minsim_all):
                    restart = True
                    break
            if restart:
                break
        
        if restart:
            # Generate first face according to orientation
            if fpj1[-1] > fpj1[0]:
                start, stop = np.sort([fpj1[0], fpj1[-1], fpi1[0], fpi1[-1]])[[0,-1]]
                fp1 = list(range(start, stop+1))
            else:
                start, stop = np.sort([fpj1[0], fpj1[-1], fpi1[0], fpi1[-1]])[[-1,0]]
                fp1 = list(range(start, stop-1, -1))
            # Generate second face according to orientation
            if fpj2[-1] > fpj2[0]:
                start, stop = np.sort([fpj2[0], fpj2[-1], fpi2[0], fpi2[-1]])[[0,-1]]
                fp2 = list(range(start, stop+1))
            else:
                start, stop = np.sort([fpj2[0], fpj2[-1], fpi2[0], fpi2[-1]])[[-1,0]]
                fp2 = list(range(start, stop-1, -1))
                    
            fp = FacePair(fp1, fp2)

            # Pop out old face pairs and place new one
            face_pair_list.pop(j)
            face_pair_list.pop(i)
            face_pair_list.append(fp)

    return face_pair_list


def crossing_boundaries(c1: Union[list,tuple,np.ndarray], c2: Union[list,tuple,np.ndarray], fp: FacePair) -> bool:
    """
    Check if the boundaries of a face pair cross.

    Args:
        c1: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 1
        c2: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of curve 2
        fp: FacePair to be checked
    
    Returns:
        True if the boundaries cross

    """
    p11, p12 = c1[fp.face1[0]], c1[fp.face1[-1]]
    p21, p22 = c2[fp.face2[0]], c2[fp.face2[-1]]
    p0 = gmt.lnr_inters(gmt.lfp(p11, p21), gmt.lfp(p12, p22))
    b1 = (p11[0] < p0[0] < p21[0]) or (p11[0] > p0[0] > p21[0])
    b2 = (p12[0] < p0[0] < p22[0]) or (p12[0] > p0[0] > p22[0])
    return b1 and b2


# GENERAL DOMAIN FUNCTIONS
def preview_mesh(dm: MeshDomain, div11 = [], div12 = [], div21 = [], div22 = []):
    """
    Meshing algorithm.

    Args:
        dm: MeshDomain. Each pair of sides must have equalamount of points.

    Returns:
        list containing all the subdomains.

    """
    # reps = 10
    # n, m = len(div11), len(div21)
    # # Calculate the division fractions for every division line. The division fractions transition linearly from one edge to another.
    # div11 = np.repeat(div11[:,np.newaxis], m, axis=0)
    # div12 = np.repeat(div12[:,np.newaxis], m, axis=0)
    # div21 = np.transpose(np.repeat(div21[:,np.newaxis], n, axis=0))
    # div22 = np.transpose(np.repeat(div22[:,np.newaxis], n, axis=0))
    # div1 = ((div12 - div11) * div22 - div12) / ((div12 - div11) * (div22 - div21) - 1)
    # div2 = ((div22 - div21) * div12 - div22) / ((div12 - div11) * (div22 - div21) - 1)
    # div2 = np.transpose(div2)
    # # Calculate boundarie conditions
    # b11 = gmt.crv_div(dm.s11, div11[0])
    # b12 = gmt.crv_div(dm.s12, div12[0])
    # b21 = gmt.crv_div(dm.s21, div21[:,0])
    # b22 = gmt.crv_div(dm.s22, div12[:,0])
    # # Calculate start conditions
    # # Converge upon the grid
    # for _ in range(reps):
    #     for segi in range(n):
    #         pass


def domain_orthosplit(dm: MeshDomain, div1 = [], div2 = []):
    """
    Split an orthogonal mesh domain. Orthogonal is the mesh domains where two sides are parallel and the other two are vertical to them and straight.

    Args:
        dm (MeshDomain): MeshDomain object
        div1 (array-like): the side length fractions of sides s11, s12 at which a the domain will be split
        div2 (array-like): the side length fractions of sides s21, s22 at which a the domain will be split

    Returns:
        list containing mesh domains
    
    Raises:
        Error: The domain doesnt have a set of parallel sides.

    """
    # Pick a side
    s11, s12 = dm.s11, dm.s12
    s21, s22 = dm.s21, dm.s22
    l1 = np.hypot(s12[:,0] - s11[:,0], s12[:,1] - s11[:,1])
    l2 = np.hypot(s22[:,0] - s21[:,0], s22[:,1] - s21[:,1])
    if np.all((l1 - np.mean(l1)) < np.mean(l1)*10**-3):
        l = np.mean(l1)
        divh, divv = div1, div2
        if gmt.vectorangle(s11[1] - s11[0], s12[0] - s11[0]) < np.pi:
            t = s11
        elif gmt.vectorangle(s11[1] - s11[0], s12[0] - s11[0]) > np.pi:
            t = s12
    elif np.all((l2 - np.mean(l2)) < np.mean(l2)*10**-3):
        l = np.mean(l2)
        divh, divv = div2, div1
        if gmt.vectorangle(s21[1] - s21[0], s22[0] - s21[0]) < np.pi:
            t = s21
        elif gmt.vectorangle(s21[1] - s21[0], s22[0] - s21[0]) > np.pi:
            t = s22

    # Prepare for division
    divh = np.sort(divh)
    divv = np.hstack(([0], np.sort(divv), [1]))
    domlist = []
    # Divide t curve
    t, j = gmt.crv_div(t, divh)
    tl = np.split(t, j, axis=0)
    for i in range(len(tl)-1):
        tl[i] = np.append(tl[i], [tl[i+1][0]], axis=0)
    # Magna-Divide
    for j in range(len(tl)):
        for i in range(1, len(divv)):
           s11 = tl[j] + gmt.parallcrv(tl[j])*divv[i-1]*l
           s12 = tl[j] + gmt.parallcrv(tl[j])*divv[i]*l
           s21 = np.array([s11[0], s12[0]])
           s22 = np.array([s11[-1], s12[-1]])
           domlist.append(MeshDomain(s11,s12,s21,s22))

    return domlist


def sb_dom_patch(dml):
    """
    Patch two neighbouring mesh domains of the same body.
    """


def control_vol(aflc, h, td):
    """
    Generate control volume curves.
    """


def trail_dom(cv, bl):
    """
    Generate the trailing curve domains.
    """


def blpatch(le_bl, te_bl):
    """
    Patch leading and trailing edge boundary layer domains into one.
    """


def edge_project(v1, v2):
    """"""


def p2p_key():
    pass


def cavity_filler():
    """
    Generate an unstructured mesh domain 
    """


def element_domain(hla: gmt.GeoShape):
    """
    Build a boundary layer domain around elements of a high lift array. 

    Args:
        hla: high lift array GeoShape

    Returns:
        ???
    
    """

# TESTING SECTIONS

