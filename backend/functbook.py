#--------- READ ME----------------------------------------------------------------
# A few function generating functions. One should refrain from editing
# already existing ones, but feel free to add new ones and use them.
#---------------------------------------------------------------------------------

import numpy as np
from scipy.interpolate import PchipInterpolator

# Function builders
def gen_pchip(y, x = None):
    """
    Generate a function by interpolating points with pchip method.

    Args:
        y (array-like): [y0, y1, ..., yn] vector containing all the y values
        x (array-like): [x0, x1, ..., xn] vector containing all the x values, if None the x values will be evenly spaced across [-1,1]

    Returns:
        f (function): function of x

    """
    y = np.array(y)
    if x == None:
        x = np.linspace(-1,1, np.shape(y)[0])
    pch = PchipInterpolator(x, y)
    funct = lambda xs: np.array([*pch(xs)])
    return funct


def gen_poly(y, x = None, deg = None):
    """
    Generate a function by fitting a polynomial onto points.

    Args:
        y (array-like): [y0, y1, ..., yn] vector containing all the y values
        x (array-like): [x0, x1, ..., xn] vector containing all the x values, if None the x values will be evenly spaced across [-1,1]
        deg (int): The degree of the polynomial, if None the greatest possible degree is picked

    Returns:
        f (function): function of x

    """
    y = np.array(y)
    if x == None:
        x = np.linspace(-1,1, np.shape(y)[0])
    if deg == None:
        deg = np.shape(y)[0]-1
    poly = np.polyfit(x,y,deg)
    funct = lambda xs: np.polyval(poly, xs)
    return funct

    """
    Generate a function of a fraction. The fraction is as shown:  c / (b + (a + xfunct(xs))*ys)**p

    Args:
        xfunct (function): function of x
        a (float): denominator multiplier
        b (float): denominator constant
        c (float): fraction numerator
        p (float): denominator power

    Returns:
        f (function): function of y,x

    """
    funct = lambda y,x : c / (b + (a + xfunct(x))*y)**p
    return funct


# Function book
