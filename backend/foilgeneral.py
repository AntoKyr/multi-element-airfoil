#--------- READ ME----------------------------------------------------------------
# THE workhorse package of this program. This library contains a number of functions
# and classes relating to manipulation and handling of airfoil points and sections.
#---------------------------------------------------------------------------------

import numpy as np
import json
import geometrics as gmt
from geomdl import fitting
from typing import Union, Callable

# AIRFOIL CLASS
class Airfoil(gmt.GeoShape):
    """
    Extends GeoShape. Describes the geometry of a single, orthodox and smooth airfoil shape.

    Attr:
        points (ndarray): Same as superclass
        squencs (list): An airfoil has only 1 (one) sequence
        shapes (list): An airfoil has only 1 (one) shape
    
    Methods:
        sides: Define the suction and pressure sides of the airfoil
        default_state: Transform the points of the airfoil, to match the usual format, of 0 AoA, and chord length of 100. Return the values related to the transformation
        camberline: Find the mean line of the airfoil
        thickness: Find the thickness of the airfoil
        le_center: Find the center of curvature of the leading edge
        hld_gen: Generate a high lift device with the Airfoil as base

    """

    def __init__(self, points: np.ndarray, squencs: list, shapes: list):
        super(Airfoil, self).__init__(points, squencs, shapes)


    def default_state(self, transform: bool = True) -> list:
        """
        Transform the points of the airfoil, to match the usual format, of 0 AoA, and chord length of 100. Return the values related to the transformation.

        Args:
            transform (bool): If True, transform the points attribute of the Airfoil. Else, only return the transformation list.

        Returns:
            list containing:
            -theta (float): the angle at which the points are rotated around [0,0]
            -scf (float): the scale factor with which the points are scaled around [0,0]
            -tv (ndarray): [x, y] the values of the translation vector
        
        """
        # Find leading edge
        points = self.points[self.squencs[0][0:-1]]
        i = np.argmax(np.hypot(points[:,0] - points[0,0], points[:,1] - points[0,1]))
        # Find transformation vectors
        transvec = -points[i]
        rottheta = gmt.vectorangle(points[0]-points[i]) 
        scalefac = 100 / np.hypot(points[i,0] - points[0,0], points[i,1] - points[0,1])

        if transform:
            # Transform
            points = gmt.translate(points, transvec)
            points = gmt.rotate(points, [0,0], -rottheta)
            self.points = gmt.scale(points, [0,0], [scalefac, scalefac])
            self.squencs[0] = list(range(np.shape(points)[0])) + [0]

        return [-rottheta, scalefac, transvec]


# SOME GENERAL FUNCTIONS
def leading_edge_indx(afl: Airfoil) -> int:
    """
    Return the index of the leading edge point of an airfoil.

    Args:
        afl: Airfoil object

    Returns:
        index

    """
    points = afl.points[afl.squencs[0]]
    return np.argmax(np.hypot(points[:,0] - points[0,0], points[:,1] - points[0,1]))


def read_ord() -> dict:
    """
    Read the airfoils.json file generated by the reform script.
    
    Returns:
        (dict): with the airfoil names as keys and the corresponding aifoil objects as values
    
    """
    with open('airfoils.json','r') as file:
        airfoils = json.load(file)
    
    aflnames = airfoils.keys()
    shapes = [[0]]
    for name in aflnames:
        ordinates = airfoils[name]
        points = np.array(ordinates)
        points[0] = (points[0] + points[-1])/2
        points = points[0:-1]
        squencs = [list(range(np.shape(points)[0])) + [0]]
        afl = Airfoil(points, squencs, shapes)
        afl.default_state()
        airfoils[name] = afl

    return airfoils


def foilpatch(le: list, te: list) -> list:
    """
    Patch leading and trailing edge of an airfoil.

    Args:
        le: The list(s) returned from the highliftdev module leading edge geometries functions
        te: The list(s) returned from the highliftdev module trailing edge geometries functions
    
    Returns:
        foil: The list(s) of the curves of the whole foil shape

    """
    el = []

    if len(le) >= 2:
        el = el + le[1:]
    if len(te) >= 2:
        el = el + te[1:]

    le = le[0]
    te = te[0]

    if len(le) > 1:
        suc = [np.vstack((te[-1], le[0]))]
        pre = [np.vstack((le[-1], te[0]))]
        le.pop(0)
        le.pop(-1)
    else:
        pre = [np.vstack((te[-1], le[0], te[0]))]
        suc = []
        le = []

    te.pop(0)
    te.pop(-1)
    foil = suc + le + pre + te
    return [foil] + el


def gs2afl(gs: gmt.GeoShape) -> Airfoil:
    """
    Check if a GeoShape meets the requirements to be an Airfoil, and if it does, return an Airfoil object with identical attributes.

    Returns:
        afl (Airfoil): Airfoil with identical attributes

    """
    # Single shape
    if len(gs.shapes) != 1:
        return False
    if len(gs.shapes[0]) != 1:
        return False
    if len(gs.squencs) != 1:
        return False
    points = gs.points[gs.squencs[0][0:-1]]
    squencs = [list(range(np.shape(points)[0]))]
    return Airfoil(points, squencs, gs.shapes)


def sides(afl: Airfoil) -> list:
    """
    Define the suction and pressure sides of the airfoil.

    Returns:
        a list containing:
        -suctside (ndarray): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the suction side
        -presside (ndarray): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the pressure side
    
    """
    points = afl.points[afl.squencs[0]]
    lei = leading_edge_indx(afl)
    suctside = points[0:lei+1]
    presside = points[lei+1:]
    return [suctside, presside]


def sides_balanced(afl: Airfoil, x: Union[list,tuple,np.ndarray] = np.linspace(1, 99, 98)) -> list:
    """
    Interpolate the airfoil's curve with spline at requested x coordinates and return the resulting sides.

    Args:
        x: the x coordinate of the ordinates
    
    Returns:
        a list containing:
        -suctside (ndarray): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the suction side
        -presside (ndarray): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the pressure side
    
    """
    xlen = len(x)
    axis = np.zeros(xlen, dtype=int)
    delta = 1 / int(1000 * xlen / (max(x) - min(x)))
    tempafl = Airfoil(afl.points, afl.squencs, afl.shapes)
    tempafl.default_state()
    interpoints = gmt.cbs_interp(tempafl.points[tempafl.squencs[0]], x, axis, [10**-3, delta])
    del tempafl
    suctside = np.append(interpoints[0:xlen], [[0,0]], axis=0)
    presside = np.insert(interpoints[xlen:], 0 , [0,0], axis=0)
    return [suctside, presside]


def sides_populate(afl: Airfoil, n: int) -> list:
    """
    Populate the airfoil's curve with ordinates at requested x coordinates.

    Args:
        x (array-like): the x coordinate of the ordinates
    
    Returns:
        afl (Airfoil): Airfoil with populated ordinates
    
    """
    le = afl.points[leading_edge_indx(afl)]
    splineafl = fitting.interpolate_curve(list(afl.points[afl.squencs[0]]), 3, centripetal=True)
    splineafl.delta = 1/n
    points = np.array(splineafl.evalpts)
    lei = max(np.argsort(np.hypot(points[:,0]-le[0], points[:,1]-le[1]))[0:2])
    suctside = np.append(points[0:lei], [le], axis=0)
    presside =points[lei:]
    return [suctside, presside]


def camberline(afl: Airfoil) -> np.ndarray:
    """
    Find the mean line of the airfoil.

    Returns:
        c (ndarray): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the curve
    """
    suctside, presside = sides_balanced(afl)
    # presside = np.insert(presside, 0, [0,0], axis=0)
    return np.vstack(([0,0], (np.flipud(suctside) + presside)/2, [100,0]))


def thickness(afl: Airfoil) -> np.ndarray:
    """
    Find the thickness of the airfoil.

    Returns:
        t (ndarray): [[x0, t0], [x1, t1], ... , [xn, tn]] the matrix containing all the thicknesses gathered, and their respective x values

    """
    suctside, presside = sides_balanced(afl)
    # presside = np.insert(presside, 0, [0,0], axis=0)
    presside[:,0] = 0
    return np.vstack(([0,0], np.flipud(suctside) - presside, [100,0]))


def le_crcl(afl: Airfoil) -> list:
    """
    Find the center and radius of curvature of the leading edge.

    Returns:
        list containing:
        -p0 (ndarray): [x, y] coordinates of the center of curvature
        -r (float): radius of curvature

    """
    # Find leading edge
    points = afl.points[afl.squencs[0][0:-1]]
    lei = leading_edge_indx(afl)
    return gmt.crcl_fit(points[lei-2:lei+3])


def hld_gen(afl: Airfoil, func: Callable, args: list) -> list:
    """
    Generate a high lift device from airfoil object.

    Args:
        func: the high lift device function as given by highliftdev module
        args: the arguments to pass to the function in a list
    
    List of functions and their arguments:
        bare_le(divx)
        le_flap1(divx, css, csp, dtheta)
        le_flap2(divx, css, csp, dtheta)
        le_flap3(divx, css, csp, dtheta)
        le_slot(divx, css, csp, cgenfunc, cgenarg, r)
        bare_te()
        te_flap(divx, cf, dtheta)
        split_flap(divx, cf, dtheta, ft)
        zap_flap(divx, cf, dtheta, ft, dx, gap, r)
        te_slot(divx, cfs, cfp, cgenfunc, cgenarg, r)
        slat(css, csp, cgenfunc, cgenarg, r)
        flap(cfs, cfp, cgenfunc, cgenarg, r)

    """
    tempafl = Airfoil(afl.points, afl.squencs, afl.shapes)
    theta, scf, tv = tempafl.default_state()
    gencurves = func(sides_populate(tempafl, 100), *args)
    del tempafl
    # Transform high lift device to original position
    for i in range(len(gencurves)):
        for j in range(len(gencurves[i])):
            curve = gencurves[i][j]
            curve = gmt.rotate(curve, [0,0], -theta)
            curve = gmt.scale(curve, [0,0], [1/scf, 1/scf])
            gencurves[i][j] = gmt.translate(curve, -tv)

    return gencurves
