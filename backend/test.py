import numpy as np
from matplotlib import pyplot as plt
import geometrics as gmt
import domain as mdd
import foilgeneral as flg
import randsectgen as rsg
import functbook
import curvegen as crvgen
import highliftdev as hld
import domain as dm
import automesher


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


if False: # big test
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


if False: # face_merge test
    plt.axis([-100, 100, -100, 100])

    curvs = draw_curves()

    c1,c2 = curvs

    dist = np.min(gmt.crv_dist(c1, c2))
    critfunc = functbook.gen_ray_crit(np.pi/4, 100*dist)

    facepairlist = mdd.opposing_faces(c1, c2, critfunc, los_check=True)

    plt.plot(c1[:,0], c1[:,1],'-')
    plt.plot(c2[:,0], c2[:,1],'-')
    plt.plot(c1[:,0], c1[:,1],'.')
    plt.plot(c2[:,0], c2[:,1],'.')

    for facepair in facepairlist:
        face1 = facepair.face1
        face2 = facepair.face2
        if mdd.crosscheck(c1, c2, facepair):
            plt.plot(c1[face1,0], c1[face1,1], '*g')
            plt.plot(c2[face2,0], c2[face2,1], '*g')
            plt.plot([c1[face1[0], 0], c2[face2[0], 0]], [c1[face1[0], 1], c2[face2[0], 1]], 'g')
            plt.plot([c1[face1[-1], 0], c2[face2[-1], 0]], [c1[face1[-1], 1], c2[face2[-1], 1]], 'g')
        else:
            plt.plot(c1[face1,0], c1[face1,1], '*c')
            plt.plot(c2[face2,0], c2[face2,1], '*c')
            plt.plot([c1[face1[0], 0], c2[face2[0], 0]], [c1[face1[0], 1], c2[face2[0], 1]], 'c')
            plt.plot([c1[face1[-1], 0], c2[face2[-1], 0]], [c1[face1[-1], 1], c2[face2[-1], 1]], 'c')

    plt.grid()
    plt.axis([-100, 100, -100, 100])

    plt.show()


if False: # cavity_id test
    plt.axis([-100, 100, -100, 100])

    curvs = draw_curves()

    c1 = curvs[0]

    plt.plot(c1[:,0], c1[:,1])
    pc = c1 - 20*gmt.parallcrv(c1)
    for i in range(len(pc)):
        plt.plot([c1[i,0], pc[i,0]], [c1[i,1], pc[i,1]], 'r')
        plt.text(c1[i,0], c1[i,1], i)

    cavities = mdd.cavity_id(c1, -20)

    for cavity in cavities:
        plt.plot(c1[cavity,0], c1[cavity,1], '*')

    plt.grid()
    plt.axis('equal')
    plt.show()


if False: # domain generation test
    tf = lambda x: 1
    plt.axis([-10, 10, -10, 10])
    gs = gmt.crv2gs([draw_curves()])[0]
    md = mdd.gs2md(gs)
    # ID a cavity
    # split the sequence
    # create a cavity domain
    # create layer domains over everything
    cavindxs = mdd.cavity_id(md.points[md.squencs[2]], 4)
    if cavindxs[-1][-1] == len(md.squencs[2]) -1:
        seqi = -1
    else:
        seqi = -2
    seqindx = md.split_sequence(2, [cavindxs[0][0], cavindxs[-1][-1]])
    md.cavity_domain(seqindx[seqi], 0.2)
    # print(md)
    print(len(md.points), len(md.spacing), len(md.squencs), len(md.nodes), len(md.shapes), len(md.mesh_types))
    gmt.gs_plot(md)


if False: # outer shell, clear duplicates test
    plt.axis([-100, 100, -100, 100])
    curvs = draw_curves()
    gs = gmt.GeoShape([],[],[])
    for curv in curvs:
        gs.add_crv(curv)
    gs.shapes = [[0,1,6],[6,7,8],[2,3,7],[4,5,8],[8,7,6]]
    gs.clear_duplicates(5)
    print(gs)
    gs.plot()
    # md = mdd.gs2md(gs)
    # print(md)
    # oush = md.outer_shell()
    # md.plot(show=False)
    # for i in oush:
    #     plt.plot(md.points[md.squencs[i], 0], md.points[md.squencs[i], 1], '*r')
    #     plt.plot(md.points[md.squencs[i], 0], md.points[md.squencs[i], 1], '--r')
    # plt.show()


if False: # shape clarify test
    plt.axis([-100, 100, -100, 100])
    curvs = draw_curves()
    gs = gmt.GeoShape([],[],[])
    for curv in curvs:
        gs.add_crv(curv)
    gs.clear_duplicates(5)
    gs.shapes = [[0,2,4,1,3]]
    print(gs)
    print(gs.shape_clarify(0, 2))


if False: # proxi point test
    plt.axis([-100, 100, -100, 100])
    curvs = draw_curves()
    gs = gmt.GeoShape([],[],[])
    for curv in curvs[:-1]:
        gs.add_crv(curv)
    gs.clear_duplicates(5)
    point = curvs[-1][0]
    pi = gs.prox_point_index(point, 2)
    gs.plot(show=False)
    x, y = gs.points[gs.squencs[2][pi]]
    plt.plot(x, y, '*')
    plt.show()


if False: # snap intersection test
    plt.axis([-100, 100, -100, 100])
    curvs = draw_curves()
    gs = gmt.GeoShape([],[],[])
    for curv in curvs[:-1]:
        gs.add_crv(curv)
    gs.clear_duplicates(5)
    coords = curvs[-1][0]
    plt.plot(coords[0], coords[1], 'o')
    for sq in gs.squencs:
        plt.plot(gs.points[sq, 0], gs.points[sq, 1], '--')
    print(coords)
    pi = gs.snap_intersection(0, coords)
    for sq in gs.squencs:
        plt.plot(gs.points[sq, 0], gs.points[sq, 1])
    plt.grid()
    plt.show()


if False: # split sequence test
    plt.axis([-100, 100, -100, 100])
    curvs = draw_curves()
    gs = gmt.GeoShape([],[],[])
    for curv in curvs:
        gs.add_crv(curv)
    gs.clear_duplicates(5)
    md = mdd.gs2md(gs)
    md.nodes = [list(np.linspace(0,1,20))]
    md.split_sequence(0, [2,6])
    print(md)


if False: # reflex domain and split hex domain test
    plt.axis([-100, 100, -100, 100])
    curvs = draw_curves()
    gs = gmt.GeoShape([],[],[])
    for curv in curvs:
        gs.add_crv(curv)
    gs.clear_duplicates(5)
    md = mdd.gs2md(gs)
    md.nodes = [[0,0.33,0.66,1], [0,0.2,0.4,0.6,0.8,1], [0,0.33,0.66,1], [0,0.2,0.4,0.6,0.8,1]]
    md.shapes = [[0, 1, 2, 3], [4,5,6,7]]
    md.mesh_types = ['hex']
    md.plot()
    md.split_hex_domain(0,0,0.3)
    print(md)
    md.plot()


if False: # opface domain test
    plt.axis([-100, 100, -100, 100])
    curvs = draw_curves()
    gs = gmt.GeoShape([],[],[])
    for curv in curvs:
        gs.add_crv(curv)
    gs.clear_duplicates(5)
    md = mdd.gs2md(gs)
    md.nodes = [[0,0.25,0.5,0.75,1], [0,0.2,0.4,0.6,0.8,1]]
    md.opface_domain(0, 1, codirectional=True, crvpn=10)
    md.plot()


if False: # attach domains test
    plt.axis([-100, 100, -100, 100])
    curvs = draw_curves()
    gs = gmt.GeoShape([],[],[])
    for curv in curvs:
        gs.add_crv(curv)
    gs.clear_duplicates(5)
    gs.shapes = [[0,1,2,3], [4,5,6,7], [5,8,9]]
    md = mdd.gs2md(gs)
    md.stitch_domains(0, 1, deform_indx=1, boundary_indx=1)
    print(md)
    md.plot()


if False: # trail domain and control volume test
    plt.axis([-100, 100, -100, 100])
    curvs = draw_curves()
    gs = gmt.GeoShape([],[],[])
    for curv in curvs:
        gs.add_crv(curv)
    gs.clear_duplicates(5)
    md = mdd.gs2md(gs)
    pi = [0,1,2,3]
    md.trailing_domain(pi, 200, functbook.gen_trail_width(0.3, 0.8), functbook.gen_sin_trail_height(70))
    md.rect_cvol(200, 70)
    print(md)
    md.plot()


if False: # swap border test
    plt.axis([-100, 100, -100, 100])
    curvs = draw_curves()
    gs = gmt.GeoShape([],[],[])
    for curv in curvs:
        gs.add_crv(curv)
    gs.clear_duplicates(5)
    gs.shapes = [[0,1,2,3]]
    md = mdd.gs2md(gs)
    md.swap_border(0, 4)
    print(md)
    md.plot()


if False: # fol_sequence test
    plt.axis([-100, 100, -100, 100])
    curvs = draw_curves()
    gs = gmt.GeoShape([],[],[])
    for curv in curvs:
        gs.add_crv(curv)
    gs.clear_duplicates(5)
    print(gs.fol_sequence(0,-1, False))


if False: # ortho_fit test
    plt.axis([-100, 100, -100, 100])
    curvs = draw_curves()
    gs = gmt.GeoShape([],[],[])
    for curv in curvs:
        gs.add_crv(curv)
    gs.clear_duplicates(5)
    gs.shapes = [[0,1,2,3], [4,5,6,7], [0,4,8]]
    md = mdd.gs2md(gs)
    md.ortho_fit(0, 1, deform_indx=1, boundary_indx=1)
    md.plot()
    print(md)


if False: # curvature test
    plt.axis([-100, 100, -100, 100])
    curv = draw_curves()[0]
    blt = 1 / -min(gmt.crv_curvature(curv))
    bl = curv + gmt.parallcrv(curv) * blt
    plt.plot(curv[:,0], curv[:,1], 'b')
    plt.plot(bl[:,0], bl[:,1], 'r')
    plt.plot(bl[:,0], bl[:,1], '.g')
    plt.axis('equal')
    plt.grid()
    plt.show()
    

if True: # test
    afld = flg.read_ord()

    _ = list(afld.keys())
    namearr = [[_[6], _[1], _[0]], [_[8], _[7], _[3]], [_[2], _[4], _[5]]]
    le_func = rsg.bare_le
    te_func = rsg.fowler_1slot

    def hld_test(name: str):
        # try:
        afl = afld[name]
        aflcrv = flg.foilpatch(le_func(afl), te_func(afl))
        gsl = gmt.crv2gs(aflcrv)
        gs = gmt.gs_merge(gsl)
        gs = dm.element_sort(gs)
        gs = dm.simple_section_prep(gs, 1, 1, 1.2)
        outers = gs.simple_boundary_layer_gen(1, 3)
        gs.simple_controlvol_gen(outers, 1500, 500, 10)
        gs.simple_trailpoint_gen(200, 300, 1500, 1, 3, 30)
        # print(gs)
        # gs.plot()
        automesher.mesh(gs, name, fin=True)
        
    hld_test(namearr[1][1])



