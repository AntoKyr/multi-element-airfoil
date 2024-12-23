import numpy as np
from matplotlib import pyplot as plt
import geometrics as gmt
import meshdomdiv as mdd
import foilgeneral as flg
import randsectgen as rsg
import curvegen as crvgen
import highliftdev as hld


def draw_crcl(p0, p1):
    r = np.linalg.norm(p1 - p0)
    thetas = np.linspace(0, 2*np.pi, 180)
    xs = r * np.cos(thetas) + p0[0]
    ys = r * np.sin(thetas) + p0[1]
    plt.plot(xs, ys)


def draw_curve(point_no, axislims = [-10,10,-10,10]):
    points = np.zeros((point_no, 2))
    fig = plt.figure()
    plt.axis(axislims)
    plt.grid()
    global drc_i
    drc_i = 0

    def cordgetter(event):
        global drc_i
        points[drc_i,0], points[drc_i,1] = event.xdata, event.ydata
        plt.plot(points[drc_i,0], points[drc_i,1], '.r')
        if drc_i > 0:
            plt.plot(points[drc_i-1:drc_i+1, 0], points[drc_i-1:drc_i+1, 1],'b')
        plt.draw()
        if drc_i == point_no - 1:
            fig.canvas.mpl_disconnect(cid)
            plt.close()
        drc_i += 1

    cid = fig.canvas.mpl_connect('button_press_event', cordgetter)
    plt.show()
    
    return points


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


if 'want to read':
    # afl = flg.read_ord(1)
    # s11 = afl.points[afl.squencs[0]]
    # s12 = s11 + 20 * gmt.parallcrv(s11)
    # s21 = np.array([s11[0], s12[0]])
    # s22 = np.array([s11[-1], s12[-1]])
    # div1 = [0.47,0.53]
    # div2 = np.linspace(0,1,4)[1:-1]
    # msl = mdd.domain_orthosplit(mdd.MeshDomain(s11,s12,s21,s22), div1, div2)
    # plt.figure()
    # for ms in msl:
    #     plt.plot(ms.s11[:,0], ms.s11[:,1], 'k')
    #     plt.plot(ms.s12[:,0], ms.s12[:,1], 'k')
    #     plt.plot(ms.s21[:,0], ms.s21[:,1], 'k')
    #     plt.plot(ms.s22[:,0], ms.s22[:,1], 'k')

    # plt.grid()
    # plt.axis('equal')
    # plt.show()


    # n = [0.5, 1.2, 3, 0.2, 2.1, 1.1, 0.7]
    # theta = [0.1, 0.3, 0.4, 0.2, 0.4, 0.3, 0.1]

    # lt = len(theta)
    # trilm = np.tril(np.ones((lt,lt)))
    # vtheta = trilm @ theta
    # vsx = n * np.sin(vtheta)
    # vsy = n * np.cos(vtheta)
    # c = np.transpose([trilm @ vsx, trilm @ vsy])
    # crvtr = gmt.crv_curvature(c)
    # print(crvtr)
    # pc = c + 1/max(crvtr) * gmt.parallcrv(c)

    # plt.plot(c[:,0], c[:,1], label='c')
    # plt.plot(pc[:,0], pc[:,1], label='pc')
    # plt.legend()
    # plt.grid()
    # plt.axis('equal')
    # plt.show()
    pass


afld = flg.read_ord()

_ = list(afld.keys())
namearr = [[_[6], _[1], _[0]], [_[8], _[7], _[3]], [_[2], _[4], _[5]]]
le_func = rsg.bare_le
te_func = rsg.bare_te

def hld_test(name: str):
    try:
        afl = afld[name]
        afl.transform([0,0], -0, [0.1, 0.1], [0, 0])
        aflcrv = flg.foilpatch(le_func(afl), te_func(afl))

        gsl = flg.crv2gs(aflcrv)
        gs = flg.gs_merge(gsl)
        flg.gs_plot(gs, indxs=False, marks=False, show=False)
        plt.title(name)
    except:
        print('*cough cough*')


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
