#--------- READ ME----------------------------------------------------------------
# A package containing the functions for the generation of mesh domains for a
# number of predetermined high lift arrays. 
#---------------------------------------------------------------------------------

import numpy as np
import geometrics as gmt

# MESH DOMAIN CLASS
class MeshDomain:
    """
    Describes a four sided mesh domain. Is composed of four geometric curves. One for each side.

    Attr:
        s11 (ndarray): First of four sides
        s12 (ndarray): Second side, opposing the first
        s21 (ndarray): Third of four sides
        s22 (ndarray): Fourth side, opposing the third

    Methods:

    """
    def __init__(self, s11, s12, s21, s22):
        self.s11 = s11
        self.s12 = s12
        self.s21 = s21
        self.s22 = s22


    def invert(self, sidekey):
        """
        Invert the sides of the domain.

        Args:
            sidekey (int): can be 1 or 2, pertaining to sides s11, s12 and s21, s22 respectively.

        """
        if sidekey == 1:
            self.s11 = np.flipud(self.s11)
            self.s12 = np.flipud(self.s12)
        elif sidekey == 2:
            self.s21 = np.flipud(self.s21)
            self.s22 = np.flipud(self.s22)


    def flip(self):
        """
        Flip the s11 and s12 with the s21 and s22.
        """
        self.s11, self.s21 = self.s21, self.s11
        self.s12, self.s22 = self.s22, self.s12



# SUPPORTING FUNCTIONS
def _ile(bodyc):
    """
    Return the index of the leading edge point of the foil.
    """
    return np.argmax(np.hypot(bodyc[:,0] - bodyc[0,0], bodyc[:,1] - bodyc[0,1]))


def _chord(bodyc):
    """
    Return the chord of the foil.
    """
    return np.sum((bodyc[_ile(bodyc)] - bodyc[0])**2)**0.5


def _lec(bodyc):
    """
    Return the circle of the leading edge.
    """
    i = _ile(bodyc)
    return gmt.crcl_fit(bodyc[i-2:i+3])


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





# MESH DOMAIN CONNECTORS



# TESTING SECTIONS

