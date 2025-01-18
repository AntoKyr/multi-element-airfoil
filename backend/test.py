import numpy as np
from matplotlib import pyplot as plt
import geometrics as gmt
import meshdomdiv as mdd
import foilgeneral as flg
import randsectgen as rsg
import functbook
import curvegen as crvgen
import highliftdev as hld


def draw_crcl(p0, p1):
    r = np.linalg.norm(p1 - p0)
    thetas = np.linspace(0, 2*np.pi, 180)
    xs = r * np.cos(thetas) + p0[0]
    ys = r * np.sin(thetas) + p0[1]
    plt.plot(xs, ys)


def draw_curves(fignum = 1):
    crv = []
    crvl = []
    fig = plt.figure(fignum)
    plt.title('Draw curves')
    plt.xlabel('[ Click: place point | Enter: finish curve | Escape: exit ]')
    plt.grid()

    def place_point(event):
        x, y = event.xdata, event.ydata
        plt.plot(x, y, '.r')
        if len(crv) > 0:
            plt.plot([crv[-1][0], x], [crv[-1][1], y],'b')
        crv.append([x, y])
        plt.draw()

    def keyvent(event):
        if event.key == 'enter':
            crvl.append(np.array(crv))
            crv.clear()
        elif event.key == 'escape':
            fig.canvas.mpl_disconnect(cid2)
            fig.canvas.mpl_disconnect(cid)
            plt.close()

    cid2 = fig.canvas.mpl_connect('key_press_event', keyvent)
    cid = fig.canvas.mpl_connect('button_press_event', place_point)

    plt.show()
    return crvl


def multivartest(array, funct):
    l1 = len(array)
    l2 = len(array[0])
    plt.figure()
    for i in range(l1):
        for j in range(l2):
            plt.subplot(l1,l2, l2*i + j + 1)
            funct(array[i][j])

    plt.tight_layout()
    plt.show()


if False: 
    afld = flg.read_ord()

    _ = list(afld.keys())
    namearr = [[_[6], _[1], _[0]], [_[8], _[7], _[3]], [_[2], _[4], _[5]]]
    le_func = rsg.act_slat
    te_func = rsg.fowler_1slot

    def hld_test(name: str):
        # try:
        afl = afld[name]
        aflcrv = flg.foilpatch(le_func(afl), te_func(afl))
        gsl = gmt.crv2gs(aflcrv)
        gs = gmt.gs_merge(gsl)
        gs = flg.element_sort(gs)
        for squence in gs.squencs:
            cf = gs.points[squence]
            bl = cf + 5 * gmt.parallcrv(cf)
            plt.plot(bl[:,0], bl[:,1], '--r')
            plt.plot([cf[0,0], bl[0,0]], [cf[0,1], bl[0,1]], '--r')
            plt.plot([cf[-1,0], bl[-1,0]], [cf[-1,1], bl[-1,1]], '--r')

        flg.gs_plot(gs, indxs=False, marks=False, show=False)
        plt.title(name)
        # except:
        #     print('*cough cough*')


    def specs(name: str):
        afl = afld[name]
        thic = max(flg.thickness(afl)[:,1])
        camb = max(flg.camberline(afl)[:,1])
        print('------------------------------------')
        print(name)
        print('Max Thickness: ' + str(thic))
        print('Max Camber: ' + str(camb))
        print('Max Camber / Max Thickness: ' + str(camb/thic))
        print('Max Camber ^ 1.5 / Max Thickness: ' + str(camb**1.5/thic))


    multivartest(namearr, hld_test)


if False:
    plt.axis([-100, 100, -100, 100])

    curvs = draw_curves()

    c1,c2 = curvs

    dist = np.min(gmt.crv_dist(c1, c2))
    critfunc = functbook.gen_ray_crit_func(np.pi/4, 5*dist)

    facepairlist = mdd.opposing_faces(c1, c2, critfunc, los_check=True)

    plt.plot(c1[:,0], c1[:,1],'-')
    plt.plot(c2[:,0], c2[:,1],'-')
    plt.plot(c1[:,0], c1[:,1],'.')
    plt.plot(c2[:,0], c2[:,1],'.')

    for facepair in facepairlist:
        face1 = facepair.face1
        face2 = facepair.face2
        plt.plot(c1[face1,0], c1[face1,1], '*c')
        plt.plot(c2[face2,0], c2[face2,1], '*c')
        plt.plot([c1[face1[0], 0], c2[face2[0], 0]], [c1[face1[0], 1], c2[face2[0], 1]], 'c')
        plt.plot([c1[face1[-1], 0], c2[face2[-1], 0]], [c1[face1[-1], 1], c2[face2[-1], 1]], 'c')

    plt.grid()
    plt.axis([-100, 100, -100, 100])

    plt.show()



