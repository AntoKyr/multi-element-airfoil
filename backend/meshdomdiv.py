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
        about a ton of them

    """
    def __init__(self, points: np.ndarray, squencs: list, shapes: list, spacing: list, nodes: list, mesh_types: list):
        super(MeshDomain, self).__init__(points, squencs, shapes)
        self.nodes = nodes
        self.spacing = spacing
        self.mesh_types = mesh_types


    # DOMAIN GENERATION
    def layer_domain(self, squenc_i: int, thickness_func: Callable[[np.ndarray], np.ndarray], mesh_type: str = 'hex') -> int:
        """
        Generate a layer domain over a sequence. 

        Args:
            squenc: the sequence index
            thickness_func: function of layer thickness over the length of the curve (length is normalized 0 <= x <= 1)
            etype: the mesh_type
        
        Returns:
            the index of the generated shape

        """
        # Prepare stuff
        points = self.points
        squenc = self.squencs[squenc_i]
        crv = points[squenc]
        # Generate stuff
        crvlen = gmt.crv_len(crv)
        lt = thickness_func(crvlen/crvlen[-1])
        layer = lt * gmt.parallcrv(crv)
        pshape, lshape = len(points), len(layer)
        squenc1 = list(range(pshape, pshape + lshape))
        squenc2 = [squenc[0], pshape]
        squenc3 = [squenc[-1], pshape + lshape]
        # Add stuff
        self.points = np.vstack((points, layer))
        self.squencs = self.squencs + [squenc1, squenc2, squenc3]
        lss = len(self.squencs)
        self.shapes.append([squenc_i, lss-2, lss-3, lss-1])
        self.mesh_types.append(mesh_type)
        # Return stuff
        return len(self.shapes) - 1


    def cavity_domain(self, squenc_c: int, bl_thickness: float) -> list:
        """
        Generate the internal domains of a cavity.

        Args:
            squenc_b: the squence that morphs the cavity
            bl_thickness: the relative boundary layer thickness: 0 < bl_thickness <= 1

        Returns:
            list containing the indexes of the generated shapes

        """
        points = self.points
        squence = self.squencs[squenc_c]
        crv = points[squence]
        cavline = np.array([crv[0], crv[-1]])
        ang1 = gmt.vectorangle(crv[-1]-crv[0], crv[1]-crv[0])
        ang2 = gmt.vectorangle(crv[0]-crv[-1], crv[-2]-crv[-1])
        thickness = 0.99 * bl_thickness / np.max(gmt.crv_curvature(crv))
        if ang1 <= np.pi * 3/4:
            gap1 = 1.5 * thickness * abs(np.cos(ang1) / np.sin(ang1))
        else:
            gap1 = 0
        if ang2 <= np.pi * 3/4:
            gap2 = 1.5 * thickness * abs(np.cos(ang2) / np.sin(ang2))
        else:
            gap2 = 0
        
        # break up sequence
        # generate sequences
        # generate shapes
        # Return stuff


    def unstruct_domain(self, squencs: list) -> int:
        """
        Generate an unstructured domain.
        """
        self.shapes.append(squencs)
        self.mesh_types.append('tri')
        return len(self.shapes) - 1


    def opface_domain(self, squenc_1: int, squenc_2: int) -> int:
        """
        Generate a structured domain from opposing face sequences.

        Args:
            squenc_1: index of first sequence
            squenc_2: index of second sequence
        
        Returns:
            index of generated shape

        """


    def control_volume(self, ):
        """
        Generate control volume.

        """


    def trailing_domain(self, ):
        """
        Generate trailing domains.
        
        """


    def patch_domains(self, shape1: int, shape2: int, squenc_1, squenc_2):
        """
        Patch neighboring domains that have a common point.

        """


    # MESH DENSITY
    def curve_noding(self, curve_i, dens_func, non: Union[int,None] = None):
        """
        """


    def point_spacing(self, point_i, spacing):
        """
        
        """


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


# MORPHOLOGY
class FacePair:
    """
    Little helper class.

    Attr:
        face1 (list): list of indexes corresponding to a curve
        face2 (list): list of indexes corresponding to another curve

    Methods:
        flip: interchange face1 and face2    
    
    """
    def __init__(self, face1: list, face2: list):
        self.face1 = face1
        self.face2 = face2


    def flip(self):
        face1, face2 = self.face1, self.face2
        self.face1, self.face2 = face2, face1


    def __str__(self):
        return 'Face 1: ' + str(self.face1) + '\nFace 2: ' + str(self.face2)


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


def face_merge(face_pair_list: list, minsim_all: float = 0, minsim_1: float = 0, minsim_2: float = 0) -> list:
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
    b1 = np.sort([p11[0], p0[0], p21[0]])
    b2 = np.sort([p11[0], p0[0], p21[0]])
    return b1 and b2


def cavity_id(c: Union[list,tuple,np.ndarray], char_len: float) -> list:
    """
    Identify cavities of a curve.

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the curve
        char_len: characteristic length, the greater it is, the less curved cavities must be to be identified.
    
    Returns:
        list with lists with indexes of cavities
    
    """
    rays = char_len * gmt.parallcrv(c) + c
    raylen = len(rays)
    blist1 = []
    blist2 = []
    # Identify all cavities
    for i in range(raylen):
        for j in range(raylen, i, -1):
            p0 = gmt.lnr_inters(gmt.lfp(c[j], rays[j]), gmt.lfp(c[i], rays[i]))
            b1 = np.sort([c[j,0], p0[0], rays[j,0]])[1] == p0[0]
            b2 = np.sort([c[i,0], p0[0], rays[i,0]])[1] == p0[0]
            if b1 and b2:
                blist1.append(i), blist2.append(j)
                break
    # Merge overlapping cavities
    blist1, blist2 = np.array(blist1), np.array(blist2)
    bi = 0
    blist1_n, blist2_n = [], []
    # Create merged cavities and note the ones that were overlapping
    while bi < len(blist1):
        b1, b2n = blist1[bi], blist2[bi]
        b2 = b2n - 1                                # just need it to be different than b2n to run the first loop
        while b2n != b2:
            b2 = b2n
            b2n = np.max(blist2[np.nonzero(blist1 <= b2)[0]])
        blist1_n.append(b1), blist2_n.append(b2)
        bi = np.nonzero(blist1 > b2)[0][0]
    # Generate curve indexes
    cavities = []
    for i in range(len(blist1_n)):
        cavities.append(list(range(blist1_n[i], blist2_n[i] + 1)))

    return cavities


# TESTING SECTIONS

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


