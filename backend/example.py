import geometrics as gmt
import foilgeneral as flg
import randsectgen as rsg
import domain as dm
import automesher

# read ordinates
afld = flg.read_ord()

# reformat all intput ordinates
_ = list(afld.keys())

# tidy them up in an array
namearr = [[_[6], _[1], _[0]], [_[8], _[7], _[3]], [_[2], _[4], _[5]]]

# select airfoil
name = namearr[1][1]
afl = afld[name]

# select le and te devices
le_func = rsg.act_slat
te_func = rsg.flap_2slot_ff

# generate le and te geometries, and merge them into a single airfoil geometry
aflcrv = flg.foilpatch(le_func(afl), te_func(afl))

# generate GeoShape objects for geometry
gsl = gmt.crv2gs(aflcrv)

# merge objects into one
gs = gmt.gs_merge(gsl)

# sort elements
gs = dm.element_sort(gs)

# prepare section for meshing
gs = dm.simple_section_prep(gs, 10, 0.95, 0.5)                   # always use as 1st
# generate higher density regions
gs.simple_trailpoint_gen(50, 70, 1500, 1.5, 0.05, 20, 80, 4)       # always use as 2nd
gs.simple_prox_layer(20, 0.5)
# generate boundary layers
outers = gs.simple_boundary_layer_gen(0.7, 50, blcoef=1.1)
# generate control volume
gs.simple_controlvol_gen(outers, 1500, 500, 60)                  # always use after boundary layer gen (obviously)

# print the domain data and plot it locally
print(gs)
gs.plot()

# generate mesh
automesher.mesh(gs, name, 3)
