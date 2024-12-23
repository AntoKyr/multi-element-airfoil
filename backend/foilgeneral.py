#--------- READ ME----------------------------------------------------------------
# THE workhorse package of this program. This library contains a number of functions
# and classes relating to manipulation and handling of airfoil points and sections.
#---------------------------------------------------------------------------------

import numpy as np
import json
from matplotlib import pyplot as plt
from matplotlib.path import Path
import geometrics as gmt
from geomdl import fitting


# GEOSHAPE CLASS
class GeoShape:
    """
    Describes a 2d geometry. Carries data relating to the points, sequences and segments of a geometric shape.

    Attr:
        points (ndarray):  [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates
        squencs (list): A list of lists of indexes pointing in the points matrix, describing the points each curve passes through
        shapes (list): A list of lists of indexes pointing in the squencs list, showing the sequences that each shape consists of

    Methods:
        transform: Transform the GeoShape's points
        recenter: Center the GeoShape, around point
    
    """

    def __init__(self, points, squencs, shapes):
        self.points = points
        self.squencs = squencs
        self.shapes = shapes


    def transform(self, center, theta, sf, tv):
        """
        Transform the GeoShape's points.

        Args:
            center (array-like): [x, y] the coordinates of the center of scaling and rotation
            theta (float): the angle of rotation
            sf (array-like): [sfx, sfy] the factors of scaling
            tv (array-like): [vx, vy] the components of displacement vector
        
        """
        points = self.points
        points = gmt.rotate(points, center, theta)
        points = gmt.scale(points, center, sf)
        self.points = gmt.translate(points, tv)


    def add_point(self, pointdata):
        """
        Adds a point at the requested squence of a GeoShape, at the requested indx.

        Args:
            pointdata (list): A list containing the data for each point to be added. Each element of the list is described bellow.
        
        Element (list containing):
            point (array-like): [x, y] coordinates of the point
            squencs (list): list containing the indexes of the curves, in which the point will be added
            indexes (list): list containing the indexes in the curves where each point will be added
        
        """


    def recenter(self, point):
        """
        Center the GeoShape, around point.

        Args:
            point (array-like): [x, y] coordinates of the point
        
        Returns:
            tv (ndarray): [x, y] coordinates of displacement vector
        
        """
        x0 = (np.max(self.points[:,0]) + np.min(self.points[:,0])) / 2
        y0 = (np.max(self.points[:,1]) + np.min(self.points[:,1])) / 2
        tv = point - np.array([x0, y0])
        self.points = gmt.translate(self.points, tv)
        return tv


# AIRFOIL CLASS
class Airfoil(GeoShape):
    """
    Subclass of GeoShape. Describes the geometry of a single, orthodox and smooth airfoil shape.

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

    def __init__(self, points, squencs, shapes):
        self.points = points
        self.squencs = squencs
        self.shapes = shapes


    def default_state(self, transform = True):
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
def read_ord():
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


def gs_merge(gsl):
    """
    Merge an array of GeoShapes, into one GeoShape object.

    Args:
        gsl (list): A list containing all the shape objects
    
    Returns:
        gs (GeoShape): A GeoShape object consisting of all others

    """
    # Assemble all the shapes
    points = np.array([], dtype=np.int64).reshape(0,2)
    squencs = []
    shapes = []
    i = 0
    j = 0
    for gs in gsl:
        points = np.vstack((points, gs.points))
        # Fix squencs indices
        gs.squencs = list(map(lambda y: list(map(lambda x: x+i, y)), gs.squencs))
        squencs = squencs + gs.squencs
        # Fix shapes indices
        gs.shapes = list(map(lambda y: list(map(lambda x: x+j, y)), gs.shapes))
        shapes = shapes + gs.shapes
        i = np.shape(points)[0]
        j = len(squencs)
    
    # Removing duplicate squencs and shapes
    # Squencs
    un_squencs = []
    inv = np.zeros(len(squencs))

    for i in range(len(squencs)):
        if squencs[i] in un_squencs:
            inv[i] = un_squencs.index(squencs[i])
        else:
            inv[i] = len(un_squencs)
            un_squencs.append(squencs[i])

    squencs = un_squencs
    for i in range(len(shapes)):
        shapes[i] = list(inv[shapes[i]])

    # Shapes
    un_shapes = []
    for i in range(len(shapes)):
        if not (shapes[i] in un_shapes):
            un_shapes.append(shapes[i])

    shapes = un_shapes
    return GeoShape(points, squencs, shapes)


def foilpatch(le, te):
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

    if np.any(le[0] != le[-1]):
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


def crv2gs(curvesls):
    """
    Generate a GeoShape from geometric curves as defined in geometrics package. 

    Args:
        curvesls (list): list of the lists containing all the geometric curves, they must have common ends and be ordered the way they connect

    Returns:
        gsl (list): List with GeoShape objects

    """
    gsl = []
    for curves in curvesls:
        j1 = 0
        squencs = []
        points = np.array([], dtype=np.int64).reshape(0,2)

        for i in range(len(curves)-1):
            points = np.vstack((points, curves[i][0:-1]))
            j2 = np.shape(points)[0]
            squencs = squencs + [list(range(j1, j2+1))]
            j1 = j2

        points = np.vstack((points, curves[-1][0:-1]))
        squencs = squencs + [list(range(j1, np.shape(points)[0])) + [0]]
        shapes = [list(range(len(curves)))]
        gsl.append(GeoShape(points, squencs, shapes))

    return gsl


def split(gs: GeoShape):
    """
    Splits a GeoShape object into all each individual shapes.

    Returns:
        shapes (list): list containing all the individual shapes as GeoShape objects, the shapes are ordered depending on their max x value, from lower to greater.
    """


def gs2afl(gs: GeoShape):
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


def cross_check(gs: GeoShape):
    """
    Check if any point of any shape in a GeoShape, is inside the polygon defined by the points of another shape of the GeoShape.
    This type of check identifies *most* crossings between shapes, and this is good enough. 

    Returns:
        bool: True if there is a "cross"

    """
    points = gs.points
    squencs = gs.squencs
    polygons = []
    for shape in gs.shapes:
        indcs = np.hstack(squencs[shape])
        polygons.append(points[indcs])

    for i in range(len(polygons)):
        for j in range(len(polygons)):
            if np.any(Path(polygons[i]).contains_points(polygons[j])):
                return True
    return False


def gs_plot(gs: GeoShape, lines: bool = True, indxs: bool = True, marks: bool = True, show: bool = True):
    """
    Plot the geometric shape.

    Args:
        lines (bool): If True, the lines of the curves will be plotted
        indxs (bool): If True, the points will be numbered according to their index in the point matrix
        marks (bool): If True, the points will be marked with dots
        show (bool): If True, the plot will be shown after plotting
    
    """
    points = gs.points
    squencs = gs.squencs

    if lines:
        for squence in squencs:
            plt.plot(points[squence,0], points[squence,1], '-b')
    
    if marks:
        plt.plot(points[:,0], points[:,1], '.r')
    
    if indxs:
        for i in range(np.shape(points)[0]):
            plt.text(points[i,0], points[i,1], i)

    plt.grid(visible=True)
    plt.axis('equal')

    if show:
        plt.show()


def sides(afl: Airfoil):
    """
    Define the suction and pressure sides of the airfoil.

    Returns:
        a list containing:
        -suctside (ndarray): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the suction side
        -presside (ndarray): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the pressure side
    
    """
    points = afl.points[afl.squencs[0]]
    lei = np.argmax(np.hypot(points[:,0] - points[0,0], points[:,1] - points[0,1]))
    suctside = points[0:lei+1]
    presside = points[lei+1:]
    return [suctside, presside]


def sides_balanced(afl: Airfoil, x = np.linspace(1, 99, 98)):
    """
    Interpolate the airfoil's curve with spline at requested x coordinates and return the resulting sides.

    Args:
        x (array-like): the x coordinate of the ordinates
    
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


def sides_populate(afl: Airfoil, n):
    """
    Populate the airfoil's curve with ordinates at requested x coordinates.

    Args:
        x (array-like): the x coordinate of the ordinates
    
    Returns:
        afl (Airfoil): Airfoil with populated ordinates
    
    """
    le = afl.points[np.argmax(np.hypot(afl.points[:,0] - afl.points[0,0], afl.points[:,1] - afl.points[0,1]))]
    splineafl = fitting.interpolate_curve(list(afl.points[afl.squencs[0]]), 3, centripetal=True)
    splineafl.delta = 1/n
    points = np.array(splineafl.evalpts)
    lei = max(np.argsort(np.hypot(points[:,0]-le[0], points[:,1]-le[1]))[0:2])
    suctside = np.append(points[0:lei], [le], axis=0)
    presside =points[lei:]
    return [suctside, presside]


def camberline(afl: Airfoil):
    """
    Find the mean line of the airfoil.

    Returns:
        c (ndarray): [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the curve
    """
    suctside, presside = sides_balanced(afl)
    # presside = np.insert(presside, 0, [0,0], axis=0)
    return (np.flipud(suctside) + presside)/2


def thickness(afl: Airfoil):
    """
    Find the thickness of the airfoil.

    Returns:
        t (ndarray): [[x0, t0], [x1, t1], ... , [xn, tn]] the matrix containing all the thicknesses gathered, and their respective x values

    """
    suctside, presside = sides_balanced(afl)
    # presside = np.insert(presside, 0, [0,0], axis=0)
    presside[:,0] = 0
    return np.flipud(suctside) - presside


def le_crcl(afl: Airfoil):
    """
    Find the center and radius of curvature of the leading edge.

    Returns:
        list containing:
        -p0 (ndarray): [x, y] coordinates of the center of curvature
        -r (float): radius of curvature

    """
    # Find leading edge
    points = afl.points[afl.squencs[0][0:-1]]
    i = np.argmax(np.hypot(points[:,0] - points[0,0], points[:,1] - points[0,1]))
    return gmt.crcl_fit(points[i-2:i+3])


def hld_gen(afl: Airfoil, func, args):
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
