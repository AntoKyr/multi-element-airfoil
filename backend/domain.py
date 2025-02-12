#--------- READ ME----------------------------------------------------------------
# A package containing the functions for the generation of mesh domains.
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


    def __str__(self):
        return super(MeshDomain, self).__str__() + '\nSpacing: ' + str(self.spacing) + '\nNodes: ' + str(self.nodes) + '\nMesh Types: ' + str(self.mesh_types)


    def remove_point(self, point_i: int) -> list:
        self.spacing.pop(point_i)
        return super(MeshDomain, self).remove_point(point_i)


    def remove_sequence(self, squence_i: int) -> list:
        self.nodes.pop(squence_i)
        return super(MeshDomain, self).remove_sequence(squence_i)


    def split_sequence(self, squence_i: int, div_i: Union[list,tuple]) -> list:
        """
        Same as super but also removes and adds relevant nodes.
        """
        sequence = self.squencs[squence_i]
        nodes = self.nodes[squence_i]
        seqlist = super(MeshDomain, self).split_sequence(squence_i, div_i)
        if nodes == None:
            self.nodes = self.nodes + list(np.full(len(seqlist), None))
        else:
            div_i = np.unique(div_i)
            if div_i[0] == 0:
                div_i = div_i[1:]
            if div_i[-1] == len(sequence):
                div_i = div_i[:-1]
            crvlen = gmt.crv_len(self.points[sequence])
            lenfrac = crvlen[div_i] / crvlen[-1]
            new_nodes = np.sort(np.concatenate((lenfrac,nodes)))
            indxs = np.concatenate(([0], np.searchsorted(nodes, lenfrac) + np.arange(0, len(lenfrac)), [len(new_nodes)-1]))
            nodelist = []
            for i in range(len(indxs)-1):
                addnodes = new_nodes[indxs[i] : indxs[i+1]+1]
                addnodes = (addnodes - addnodes[0]) / (addnodes[-1] - addnodes[0]) 
                nodelist.append(addnodes)
            self.nodes = self.nodes + nodelist

        return seqlist


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
        lt = thickness_func(crvlen / crvlen[-1])
        layer = crv + lt * gmt.parallcrv(crv)
        pshape, lshape = len(points), len(layer)
        squenc1 = list(range(pshape, pshape + lshape))
        squenc2 = [squenc[0], pshape]
        squenc3 = [squenc[-1], pshape + lshape - 1]
        # Add stuff
        self.points = np.vstack((points, layer))
        self.squencs = self.squencs + [squenc1, squenc2, squenc3]
        lss = len(self.squencs)
        self.shapes.append([squenc_i, lss-2, lss-3, lss-1])
        self.mesh_types.append(mesh_type)
        self.spacing = self.spacing + list(np.full(len(crvlen), None))
        self.nodes = self.nodes + [self.nodes[squenc_i], None, None]
        # Return stuff
        return len(self.shapes) - 1


    def cavity_domain(self, squenc_c: int, bl_thickness: float = 0.1) -> list:
        """
        Generate the internal domains of a cavity.

        Args:
            squenc_b: the squence that morphs the cavity
            bl_thickness: the relative boundary layer thickness: 0 < bl_thickness < 1

        Returns:
            list containing the indexes of the generated shapes

        """
        points = self.points
        squence = self.squencs[squenc_c]
        crv = points[squence]
        self.squencs.append([squence[0], squence[-1]])
        self.nodes.append(None)
        ang1 = gmt.vectorangle(crv[1]-crv[0], crv[-1]-crv[0])
        ang2 = gmt.vectorangle(crv[0]-crv[-1], crv[-2]-crv[-1])
        thickness = bl_thickness / np.max(np.abs(gmt.crv_curvature(crv)))
        if ang1 <= np.pi * 3/4:
            gap1 = 1.5 * thickness * abs(np.cos(ang1) / np.sin(ang1))
        else:
            gap1 = 0
        if ang2 <= np.pi * 3/4:
            gap2 = 1.5 * thickness * abs(np.cos(ang2) / np.sin(ang2))
        else:
            gap2 = 0
        crvlen = gmt.crv_len(crv)
        i1 = np.nonzero(crvlen >= gap1)[0][0]
        i2 = np.nonzero(crvlen <= crvlen[-1] - gap2)[0][-1]
        self.split_sequence(squenc_c, [i1,i2])
        self.layer_domain(len(self.squencs)-2, lambda x: thickness)
        sl = len(self.squencs)
        self.unstruct_domain([sl-1,sl-2,sl-3,sl-4,sl-7,sl-6])
        sl = len(self.shapes)
        return [sl-2, sl-1]


    def unstruct_domain(self, squencs: list) -> int:
        """
        Generate an unstructured domain.
        """
        self.shapes.append(squencs)
        self.mesh_types.append('tri')
        return len(self.shapes) - 1


    def opface_domain(self, squenc_1: int, squenc_2: int, codirectional: bool = True, minang = np.pi/3, crvpn: int = 2) -> int:
        """
        Generate a structured domain from opposing face sequences.

        Args:
            squenc_1: index of first sequence
            squenc_2: index of second sequence
            codirectional: True if both curves are generally co-directional
            minang: the minimum possible angle of the "vertical" to the faces, domain curves
            crvpn: the number of points of the "vertical" curves
        
        Returns:
            index of generated shape

        """
        points = self.points
        squencs = self.squencs
        if codirectional:
            i2 = 0
        else:
            i2 = -1
        i1s1 = squencs[squenc_1][0]
        i1s2 = squencs[squenc_1][1]
        i1e1 = squencs[squenc_1][-1]
        i1e2 = squencs[squenc_1][-2]
        i2s1 = squencs[squenc_2][i2]
        i2s2 = squencs[squenc_2][gmt.nxt_i(i2)]
        i2e1 = squencs[squenc_2][gmt.opst_i(i2)]
        i2e2 = squencs[squenc_2][gmt.nxt_i(gmt.opst_i(i2))]

        ang1s = gmt.vectorangle(points[i1s2] - points[i1s1], points[i2s1] - points[i1s1])
        ang2s = gmt.vectorangle(points[i2s2] - points[i2s1], points[i1s1] - points[i2s1])
        ang1e = gmt.vectorangle(points[i1e2] - points[i1e1], points[i2e1] - points[i1e1])
        ang2e = gmt.vectorangle(points[i2e2] - points[i2e1], points[i1e1] - points[i2e1])

        if (abs(ang1s) < minang) or (abs(ang2s) < minang):
            p1 = points[i1s1]
            p2 = points[i2s1]
            if (abs(ang1s) < minang):
                p2 = gmt.rotate(p2, p1, -np.sign(ang1s) * (minang - abs(ang1s)))
            else:
                p2 = gmt.rotate(p2, p1, -np.sign(ang1s) * (minang - abs(ang2s)) / 2)
            lf1 = gmt.lfp(p1,p2)
            p2 = points[i2s1]
            if (abs(ang2s) < minang):
                p1 = gmt.rotate(p1, p2, -np.sign(ang2s) * (minang - abs(ang2s)))
            else:
                p1 = gmt.rotate(p1, p2, -np.sign(ang2s) * (minang - abs(ang1s)) / 2)
            lf2 = gmt.lfp(p1,p2)
            p0s = gmt.lnr_inters(lf1, lf2)
        else:
            p0s = (points[i1s1] + points[i2s1]) / 2

        if (abs(ang1e) < minang) or (abs(ang2e) < minang):
            p1 = points[i1e1]
            p2 = points[i2e1]
            if (abs(ang1e) < minang):
                p2 = gmt.rotate(p2, p1, -np.sign(ang1e) * (minang - abs(ang1e)))
            else:
                p2 = gmt.rotate(p2, p1, -np.sign(ang1e) * (minang - abs(ang2e)) / 2)
            lf1 = gmt.lfp(p1,p2)
            p2 = points[i2e1]
            if (abs(ang2e) < minang):
                p1 = gmt.rotate(p1, p2, -np.sign(ang2e) * (minang - abs(ang2e)))
            else:
                p1 = gmt.rotate(p1, p2, -np.sign(ang2e) * (minang - abs(ang1e)) / 2)
            lf2 = gmt.lfp(p1,p2)
            p0e = gmt.lnr_inters(lf1, lf2)
        else:
            p0e = (points[i1s1] + points[i2s1]) / 2

        vcs = gmt.bezier([points[i1s1], p0s, points[i2s1]])(np.linspace(0,1,crvpn))[1:-1]
        vce = gmt.bezier([points[i1e1], p0e, points[i2e1]])(np.linspace(0,1,crvpn))[1:-1]
        
        l1, l2, l3 = len(points), len(vcs), len(vce)

        sq1 = [i1s1] + list(range(l1, l1 + l2)) + [i2s1]
        sq2 = [i1e1] + list(range(l1 + l2, l1 + l2 + l3)) + [i2e1]

        ls = len(self.squencs)
        shp = [squenc_1, ls, squenc_2, ls + 1]

        self.points = np.vstack((points, vcs, vce))
        self.squencs = self.squencs + [sq1, sq2]
        self.shapes.append(shp)
        self.mesh_types.append('hex')


    def reflex_domain(self, sq1: int, sq2: int):
        """
        Create a domain on two sequences with reflex angle. The sequences must have a common end point.

        Args:
            squenc_1: index of first sequence
            squenc_2: index of second sequence

        """
        points = self.points
        sequence_1 = self.squencs[sq1]
        sequence_2 = self.squencs[sq2]
        if sequence_1[-1] == sequence_2[0]:
            trv1 = points[sequence_2[-1]] - points[sequence_2[0]]
            trv2 = points[sequence_1[0]] - points[sequence_1[-1]]
            seqslice = np.s_[0:-1]
            i1 = len(sequence_1) - 1
            i2 = -1
        elif sequence_1[-1] == sequence_2[-1]:
            trv1 = points[sequence_2[0]] - points[sequence_2[-1]]
            trv2 = points[sequence_1[0]] - points[sequence_1[-1]]
            seqslice = np.s_[0:-1]
            i1 = len(sequence_1) - 1
            i2 = 0
        elif sequence_1[0] == sequence_2[0]:
            trv1 = points[sequence_2[-1]] - points[sequence_2[0]]
            trv2 = points[sequence_1[-1]] - points[sequence_1[0]]
            seqslice = np.s_[1:]
            i1 = 0
            i2 = -1
        elif sequence_1[0] == sequence_2[-1]:
            trv1 = points[sequence_2[0]] - points[sequence_2[-1]]
            trv2 = points[sequence_1[-1]] - points[sequence_1[0]]
            seqslice = np.s_[1:]
            i1, i2 = 0, 0

        self.add_crv(gmt.translate(points[sequence_1][seqslice], trv1))
        self.nodes.append(self.nodes[sq1])
        self.squencs[-1].insert(i1, sequence_2[i2])

        self.add_crv(gmt.translate(points[sequence_2][1:-1], trv2))
        self.nodes.append(self.nodes[sq2])
        self.squencs[-1].insert((1+i2) * len(sequence_2), sequence_1[gmt.opst_i(i1)])
        self.squencs[-1].insert(-i2 * len(sequence_2), self.squencs[-2][gmt.opst_i(i1)])
        i = len(self.squencs) - 1
        self.shapes.append([i, sq1, i - 1, sq2])
        self.mesh_types.append('hex')


    def rect_cvol(self, x, h):
        """
        Generate rectangular control volume.

        Args:
            x: the distance of the trailing domain boundary
            h: the distance of the upper domain boundary

        """
        outer_shell = self.outer_shell()
        trail_pi = []
        i = 0
        while i < len(outer_shell):
            x1 = self.points[self.squencs[outer_shell[i]][0],0]
            x2 = self.points[self.squencs[outer_shell[i]][-1],0]
            if (x1 >= x) and (x2 >= x):
                trail_pi = trail_pi + [self.squencs[outer_shell[i]][0], self.squencs[outer_shell[i]][-1]]
                outer_shell.pop(i)
            else:
                i += 1

        trail_pi = list(set(trail_pi))
        maxpi = trail_pi[np.argmax(self.points[trail_pi, 1])]
        minpi = trail_pi[np.argmin(self.points[trail_pi, 1])]
        lp = len(self.points)
        self.points = np.vstack((self.points, [[x,h], [-h,h], [-h,-h], [x,-h]]))
        ls = len(self.squencs)
        self.squencs = self.squencs + [[maxpi, lp], [lp, lp+1], [lp+1, lp+2], [lp+2, lp+3], [lp+3, minpi]]
        self.nodes = self.nodes + [None, None, None, None, None]
        self.shapes.append(outer_shell + [ls, ls+1, ls+2, ls+3, ls+4])
        self.mesh_types.append('tri')


    def trailing_domain(self, tp: list, x: float, trail_width_func: Callable[[np.ndarray], np.ndarray], trail_height_func: Callable[[float], float]):
        """
        Generate trailing domains.

        Args:
            tp: contains all point indexes of the furthest back domains, that the trailing domains are attached to
            x: the x coordinate that the trail mesh domain stretches out to
            trail_width_func: influences the width of the trail depending on the length of the trail
            trail_height_func: influences the height of the trail depending on the angle of the trailing domains 

        """
        points = self.points
        # get mean trail vector
        unvec = []
        for pi in tp:
            reflist = self.point_ref(pi)
            for refseq in reflist:
                indx = self.squencs[refseq].index(pi)
                vct = points[pi] - points[self.squencs[refseq][gmt.nxt_i(indx)]]
                unvct = vct / np.linalg.norm(vct)
                unvec.append(unvct)
        unvec = np.sum(unvec, axis=0)
        # get the relative point positions
        p1 = points[tp]
        topi = np.argmax(p1[:,1])
        boti = np.argmin(p1[:,1])
        maxd = np.hypot(points[topi,0]-points[boti,0], points[topi,1]-points[boti,1])
        lf = gmt.lfp(points[topi], points[boti])
        projp = np.zeros((len(p1), 2))
        for i in range(len(p1)):
            projp[i] = gmt.project(p1[i], lf)

        ppd = np.hypot(projp[:,0] - points[boti,0], projp[:,1] - points[boti,1]) - maxd/2

        # place far away points of trail
        mpy = (points[boti,1] + points[topi,1])/2
        mp2 = [x, trail_height_func(gmt.vectorangle(unvec)) + mpy]
        p2 = np.transpose([x * np.ones(len(ppd)), mp2[1] + ppd])

        # create curves
        curves = []
        if -10**-4 * maxd < trail_height_func(gmt.vectorangle(unvec)) < 10**-4 * maxd: # give some tolerance
            for i in range(len(ppd)):
                curves.append(np.transpose([np.linspace(p1[i,0], p2[i,0], 30), p1[i,1]]))
        else:
            for i in range(len(ppd)):
                pint = gmt.lnr_inters(gmt.lfp(p1[i], p1[i]+unvec), [0, p2[i,1]])
                curves.append(gmt.bezier([p1[i], pint, p2[i]])(np.linspace(0,1,30)))
        
        # add width
        for i in range(len(curves)):
            curves[i] = curves[i] + trail_width_func(curves[i]) * 2 * ppd[i] / maxd
            curves[i][-1,0] = x  # fix sligh x displacement for consistent results
        
        # add curves
        ls = len(self.squencs)
        sqis = []
        for i in range(len(tp)):
            self.add_crv(curves[i][1:])
            self.squencs[-1].insert(0, tp[i])
            sqis.append(ls + i)

        # create shapes and add them
        srtis = np.argsort(ppd)
        for i in range(len(srtis)-1):
            sqi2 = self.multipoint_ref([tp[srtis[i]], tp[srtis[i+1]]])[0]
            sqi1 = sqis[srtis[i]]
            sqi3 = sqis[srtis[i+1]]
            self.squencs.append([self.squencs[sqi1][-1], self.squencs[sqi3][-1]])
            self.nodes.append(self.nodes[sqi2])
            self.shapes.append([sqi1, sqi2, sqi3, len(self.squencs)-1])
            self.mesh_types.append('hex')


    # DOMAIN MANIPULATION
    def swap_border(self, shape_i: int, squenc_b: int, squenc_i: int = None):
        """
        Swap a domain's old border sequence with a new one.

        Args:
            shape_i: the index of the shape
            squenc_b: the index of the sequence that will replace the old one
            squenc_i: the index of the sequence that will be replaced, if not given the most suitable will be picked

        """
        points = self.points
        squencs = self.squencs
        shapes = self.shapes
        # Get all viable sequences, if not given as argument
        if squenc_i == None:
            seqis = list(shapes[shape_i])
            i = 0
            while i < len(seqis):
                if len(self.sequence_ref(seqis[i])) > 1:
                    seqis.pop(i)
                else:
                    i += 1
        else:
            seqis = [squenc_i]
        
        # Gather sequence combinations
        seq_dists = np.zeros(len(seqis))
        p11 = points[squencs[squenc_b][0]]
        p12 = points[squencs[squenc_b][-1]]
        for i in range(len(seqis)):
            p21 = points[squencs[seqis[i]][0]]
            p22 = points[squencs[seqis[i]][-1]]
            tmpdist1 = np.hypot(p11[0] - p21[0], p11[1] - p21[1])
            tmpdist2 = np.hypot(p12[0] - p22[0], p12[1] - p22[1])
            dist1 = np.hypot(tmpdist1, tmpdist2)
            tmpdist1 = np.hypot(p11[0] - p22[0], p11[1] - p22[1])
            tmpdist2 = np.hypot(p12[0] - p21[0], p12[1] - p21[1])
            dist2 = np.hypot(tmpdist1, tmpdist2)
            seq_dists[i] = min(dist1, dist2)

        # Get best fitting combination
        i = np.argmin(seq_dists)
        sq1 = squencs[seqis[i]]
        sq2 = squencs[squenc_b]
        crv1 = points[sq1]
        crv2 = points[sq2]

        # Check if flip is needed
        tmpdist1 = np.hypot(crv1[0,0] - crv2[0,0], crv1[0,1] - crv2[0,1])
        tmpdist2 = np.hypot(crv1[-1,0] - crv2[-1,0], crv1[-1,1] - crv2[-1,1])
        dist1 = np.hypot(tmpdist1, tmpdist2)
        tmpdist1 = np.hypot(crv1[0,0] - crv2[-1,0], crv1[0,1] - crv2[-1,1])
        tmpdist2 = np.hypot(crv1[-1,0] - crv2[0,0], crv1[-1,1] - crv2[0,1])
        dist2 = np.hypot(tmpdist1, tmpdist2)
        if dist1 > dist2:
            i0 = sq2[-1]
            i1 = sq2[0]
        else:
            i0 = sq2[0]
            i1 = sq2[-1]
        
        self.snap_intersection(sq1[0], points[i0])
        self.snap_intersection(sq1[-1], points[i1])

        # replace stuff
        pi0, pi1 = sq1[0], sq1[-1]
        self.replace_point(pi0, i0)
        self.replace_point(pi1, i1)
        self.replace_sequence(seqis[i], squenc_b, [shape_i])

        # clean up
        if pi0 == i0:
            self.squencs[seqis[i]].pop(0)
        else:
            self.squencs[seqis[i]][0] = pi0
        if pi1 == i1:
            self.squencs[seqis[i]].pop(-1)
        else:
            self.squencs[seqis[i]][-1] = pi1
        for j in range(len(self.squencs[seqis[i]])):
            self.remove_point(self.squencs[seqis[i]][0])
        self.remove_sequence(seqis[i])


    def stitch_domains(self, shape_1: int, shape_2: int, squenc_1: int = None, squenc_2: int = None, deform_indx: int = 0, boundary_indx: int = 0):
        """
        Attach two "parallel" domains by deforming them as needed and merging their borders.

        Args:
            shape_1: the index of the first shape
            shape_2: the index of the second shape
            squenc_1: the index of the first shape sequence that will be patched, if not given the most suitable will be picked
            squenc_2: the index of the second shape sequence that will be patched, if not given the most suitable will be picked
            deform_indx: if 0, both domains are deformed, if 1, only the second domain is deformed, if 2, only the first is
            boundary_indx: if 0, both boundaries are merged into a mean, if 1 only the first boundary is used, if 2, only the second

        """
        points = self.points
        squencs = self.squencs
        shapes = self.shapes
        # Get all viable sequences, if not given as arguments
        if squenc_1 == None:
            seqis_1 = list(shapes[shape_1])
            i = 0
            while i < len(seqis_1):
                if len(self.sequence_ref(seqis_1[i])) > 1:
                    seqis_1.pop(i)
                else:
                    i += 1
        else:
            seqis_1 = [squenc_1]

        if squenc_2 == None:
            seqis_2 = list(shapes[shape_2])
            i = 0
            while i < len(seqis_2):
                if len(self.sequence_ref(seqis_2[i])) > 1:
                    seqis_2.pop(i)
                else:
                    i += 1
        else:
            seqis_2 = [squenc_2]

        # Gather sequence combinations
        seq_dists = np.zeros((len(seqis_1), len(seqis_2)))
        for i1 in range(len(seqis_1)):
            p11 = points[squencs[seqis_1[i1]][0]]
            p12 = points[squencs[seqis_1[i1]][-1]]
            for i2 in range(len(seqis_2)):
                p21 = points[squencs[seqis_2[i2]][0]]
                p22 = points[squencs[seqis_2[i2]][-1]]
                tmpdist1 = np.hypot(p11[0] - p21[0], p11[1] - p21[1])
                tmpdist2 = np.hypot(p12[0] - p22[0], p12[1] - p22[1])
                dist1 = np.hypot(tmpdist1, tmpdist2)
                tmpdist1 = np.hypot(p11[0] - p22[0], p11[1] - p22[1])
                tmpdist2 = np.hypot(p12[0] - p21[0], p12[1] - p21[1])
                dist2 = np.hypot(tmpdist1, tmpdist2)
                seq_dists[i1, i2] = min(dist1, dist2)

        # Get best fitting combination
        i1 = np.argmin(np.min(seq_dists, axis=1))
        i2 = np.argmin(np.min(seq_dists, axis=0))
        sq1 = squencs[seqis_1[i1]]
        sq2 = squencs[seqis_2[i2]]
        crv1 = points[sq1]
        crv2 = points[sq2]

        # Check and flip if needed
        tmpdist1 = np.hypot(crv1[0,0] - crv2[0,0], crv1[0,1] - crv2[0,1])
        tmpdist2 = np.hypot(crv1[-1,0] - crv2[-1,0], crv1[-1,1] - crv2[-1,1])
        dist1 = np.hypot(tmpdist1, tmpdist2)
        tmpdist1 = np.hypot(crv1[0,0] - crv2[-1,0], crv1[0,1] - crv2[-1,1])
        tmpdist2 = np.hypot(crv1[-1,0] - crv2[0,0], crv1[-1,1] - crv2[0,1])
        dist2 = np.hypot(tmpdist1, tmpdist2)
        flipindx = 0
        if dist1 > dist2:
            crv2 = np.flipud(crv2)
            flipindx = -1
        
        # generate merged boundary
        # boundary edge points
        bps = [[(crv1[0]+crv2[0])/2, (crv1[-1]+crv2[-1])/2], [crv1[0], crv1[-1]], [crv2[0], crv2[-1]]]
        bp = bps[deform_indx]
        # boundary curve
        crvbs = [gmt.mean_crv(crv1, crv2), crv1, crv2]
        crvb = crvbs[boundary_indx]
        # merged curve
        crvm = gmt.crv_fit2p(crvb, bp[0], bp[1])

        # Check for common ends
        common_end1 = sq1[0] == sq2[flipindx]
        common_end2 = sq1[-1] == sq2[gmt.opst_i(flipindx)]

        # Snap all conected curves to new points, if needed
        if not common_end1:
            psnap = crvm[0]
            pi1s = sq1[0]
            pi2s = sq2[flipindx]
            self.snap_intersection(pi1s, psnap)
            self.snap_intersection(pi2s, psnap)
        if not common_end2:
            psnap = crvm[-1]
            pi1e = sq1[-1]
            pi2e = sq2[gmt.opst_i(flipindx)]
            self.snap_intersection(pi1e, psnap)
            self.snap_intersection(pi2e, psnap)

        # Add new sequence
        squencs.append(list(range(len(points), len(points) + len(crvm))))
        points = np.vstack((points, crvm))
        self.points = points
        self.squencs = squencs

        # Take note of points to remove after replacing
        remi = [sq1[0], sq1[-1], sq2[flipindx], sq2[gmt.opst_i(flipindx)]]

        # Replace start and end points with new sequence start and end
        self.replace_point(sq1[0], self.squencs[-1][0])
        self.replace_point(sq1[-1], self.squencs[-1][-1])
        self.replace_point(sq2[flipindx], self.squencs[-1][0])
        self.replace_point(sq2[gmt.opst_i(flipindx)], self.squencs[-1][-1])

        # remove edge points
        self.squencs.insert(0,remi)
        while len(self.squencs[0]) > 0:
            self.remove_point(self.squencs[0][0])
        self.squencs.pop(0)

        # create shape with new sequences to keep track of them because the code is spaghetti
        self.shapes.insert(0, [seqis_1[i1], seqis_2[i2]])

        # Remove all old sequence points except the start and end points
        for sq in self.shapes[0]:
            while len(self.squencs[sq]) > 2:
                self.remove_point(self.squencs[sq][1])

        # Get max node number sequence and use these nodes for new sequence
        if (self.nodes[seqis_1[i1]] != None) and (self.nodes[seqis_2[i2]] != None):
            nn = np.argmax([len(self.nodes[seqis_1[i1]]), len(self.nodes[seqis_2[i2]])])
            new_nodes = [self.nodes[seqis_1[i1]], self.nodes[seqis_2[i2]]][nn]
        elif self.nodes[seqis_1[i1]] != None:
            new_nodes = self.nodes[seqis_1[i1]]
        elif self.nodes[seqis_2[i2]] != None:
            new_nodes = self.nodes[seqis_1[i1]]
        else:
            new_nodes = None

        self.nodes.append(new_nodes)

        # Remove curves and place new sequence into shapes
        seqi = len(self.squencs) - 3
        shi1, inserindxs1 = self.remove_sequence(self.shapes[0][0])
        shi2, inserindxs2 = self.remove_sequence(self.shapes[0][0])
        shi = shi1 + shi2
        insertindxs = inserindxs1 + inserindxs2
        for i in range(len(shi)):
            self.shapes[shi[i]].insert(insertindxs[i], seqi)
        self.shapes.pop(0)


    def split_hex_domain(self, shape_i: int, splt_seq: int, splt_fract: float):
        """
        Split a hex domain into two.

        Args:
            shape_i: the index of the shape to be split
            splt_seq: the within-shape index
            split_i: the fraction of length at which the split occurs

        """
        points = self.points
        squencs = self.squencs
        shape = self.shapes[shape_i]
        # figure it out 
        order, orient = self.shape_clarify(shape_i, splt_seq)
        sqs1 = shape[splt_seq]
        sqs2 = shape[order[2]]
        sqv1 = shape[order[1]]
        sqv2 = shape[order[3]]

        sequence_s1 = squencs[sqs1]
        sequence_s2 = squencs[sqs2]
        sequence_v1 = squencs[sqv1]
        sequence_v2 = squencs[sqv2]
        if orient[2] == 0:
            sequence_s2.reverse()
            if self.nodes[sqs2] != None:
                self.nodes[sqs2].reverse()
        if orient[1] == -1:
            sequence_v1.reverse()
            if self.nodes[sqv1] != None:
                self.nodes[sqv1].reverse()
        if orient[3] == 0:
            sequence_v2.reverse()
            if self.nodes[sqv2] != None:
                self.nodes[sqv2].reverse()

        crv1len = gmt.crv_len(points[squencs[sqs1]])
        i1 = np.argmin(np.abs(crv1len/crv1len[-1] - splt_fract))
        if (i1 == 0):
            i1 = 1
        elif (i1 == (len(crv1len) - 1)):
            i1 = i1 - 1
        
        crv2len = gmt.crv_len(points[squencs[sqs1]])
        i2 = np.argmin(np.abs(crv2len/crv2len[-1] - splt_fract))
        if (i2 == 0):
            i2 = 1
        elif (i2 == (len(crv2len) - 1)):
            i2 = i2 - 1
        
        # get vertical sequence
        crvv1 = gmt.crv_fit2p(points[squencs[sqv1]], points[sequence_s1[i1]], points[sequence_s2[i2]])
        crvv2 = gmt.crv_fit2p(points[squencs[sqv2]], points[sequence_s1[i1]], points[sequence_s2[i2]])
        sqv0 = gmt.mean_crv(crvv2, crvv1, 1 - splt_fract)[1:-1]
        self.add_crv(sqv0)
        self.nodes.append(self.nodes[sqv1])
        self.squencs[-1].insert(0, sequence_s1[i1])
        self.squencs[-1].append(sequence_s2[i2])

        # split sequences
        if sqs1 > sqs2:
            is1 = 0
        else:
            is1 = -1

        # create fake shapes to keep track of where the fuck the sequences move because my code is so spaghetti i wanna pull my eyes out, but i dont have time to fix it, fuuuuuuuuuuuuuuuuuuuu
        self.shapes.insert(0, [sqs2])
        self.shapes.insert(0, [sqv1, sqv2])

        self.split_sequence(sqs1, [i1])
        self.split_sequence(self.shapes[1][0], [i2])

        sqv1 = self.shapes[0][0]
        sqv2 = self.shapes[0][1]

        # remove fake shapes
        self.shapes.pop(0), self.shapes.pop(0)

        # create shapes
        self.shapes.pop(shape_i)
        si = len(self.squencs) - 1
        shape1 = [sqv2, si-3, si-4, si-1]
        shape2 = [sqv1, si-2, si-4, si]
        self.shapes.insert(shape_i, shape1)      # we put this here to not mess with shape indexing
        self.shapes.append(shape2)

        # mesh types
        self.mesh_types.pop(shape_i)
        self.mesh_types = self.mesh_types + ['hex','hex']


    def ortho_fit(self, shape_v: int, shape_s: int, squenc_v: int = None, squenc_s: int = None, deform_indx: int = 0, boundary_indx: int = 0) -> bool:
        """
        Attach two orthogonal hex domains by splitting one, and attaching one part to the other. Both domains must have one common vertex point for this to work.

        Args:
            shape_v: the index of the first shape, this shape wont be split
            shape_s: the index of the second shape, this shape will be split
            squenc_1: the index (within the shape) of the first shape sequence that will be patched, if not given the most suitable will be picked
            squenc_2: the index (within the shape) of the second shape sequence that will be patched, if not given the most suitable will be picked
            boundary_indx: if 0, both boundaries are merged into a mean, if 1 only the first boundary is used, if 2, only the boundary of the split domaon is used
        
        Returns:
            True if the fit succeeds

        """
        points = self.points
        squencs = self.squencs
        shapes = self.shapes
        shs = shapes[shape_s]
        shv = shapes[shape_v]

        # Get all vertexes and find common
        order_sqs_v, orient_v = self.shape_clarify(shape_v)
        order_sqs_s, orient_s = self.shape_clarify(shape_s)
        vertex_v, vertex_s = [], []
        for i in range(len(order_sqs_v)):
            vertex_v.append(squencs[shv[order_sqs_v[i]]][orient_v[i]])
        for i in range(len(order_sqs_s)):
            vertex_s.append(squencs[shs[order_sqs_v[i]]][orient_s[i]])
        for vv in vertex_v:
            if vv in vertex_s:
                cvi = vv

        # construct polygon
        polyg = np.array([], dtype=np.int64).reshape(0,2)
        for i in range(len(order_sqs_s)):
            curve = points[squencs[shs[order_sqs_s[i]]]]
            if orient_s[i] == -1:
                curve = np.flipud(curve)
                polyg = np.vstack((polyg, curve))

        # check failure conditions
        # if more than one vertex inside shape_s polygon: fail
        vertex_v.remove(cvi)
        vertin = []
        for vv in vertex_v:
            if gmt.inpolyg(points[vv], polyg):
                vertin.append(vv)
        if len(vertin) > 1:
            return False
        # if both or no sequences that have the common point, are also used in other shapes, without the sequence being given in args: fail
        if squenc_v == None:
            seqis_v = self.point_ref(cvi, shv)
            i = 0
            while i < len(seqis_v):
                if len(self.sequence_ref(seqis_v[i])) > 1:
                    seqis_v.pop(i)
                else:
                    i += 1
            if len(seqis_v) != 1:
                return False
            else:
                seqi_v = seqis_v[0]
        else:
            seqi_v = squenc_v

        if squenc_s == None:
            seqis_s = self.point_ref(cvi, shs)
            i = 0
            while i < len(seqis_s):
                if len(self.sequence_ref(seqis_s[i])) > 1:
                    seqis_s.pop(i)
                else:
                    i += 1
            if len(seqis_s) != 1:
                return False
            else:
                seqi_s = seqis_s[0]
        else:
            seqi_s = squenc_s

        # get closest point indx and calculate split fraction
        edgeindx = gmt.opst_i(squencs[seqi_v].index(cvi))
        proxindx = self.prox_point_index(points[squencs[seqi_v]][edgeindx], seqi_s)
        crvlen = gmt.crv_len(points[squencs[seqi_s]])
        splt_fract = crvlen[proxindx] / crvlen[-1]

        # find sequence orientation
        cvi_sindx = squencs[seqi_s].index(cvi)
        if cvi_sindx == 0:
            patch_i = shape_s
        else:
            patch_i = len(shapes)

        # cut n stitch
        self.split_hex_domain(shape_s, shs.index(seqi_s), splt_fract)
        self.stitch_domains(shape_v, patch_i, deform_indx=deform_indx, boundary_indx=boundary_indx)

        return True


    # MESH DENSITY
    def curve_noding(self, curve_i, dens_func, non: Union[int,None] = None):
        """
        """


    def point_spacing(self, point_i, spacing):
        """
        
        """


def gs2md(gs: gmt.GeoShape) -> MeshDomain:
    """
    Create a MeshDomain from a GeoShape.
    
    Args:
        gs: GeoShape to be used

    Returns:
        MeshDomain from GeoShape

    """
    mesh_types = list(np.full(len(gs.shapes), None))
    nodes = list(np.full(len(gs.squencs), None))
    spacing = list(np.full(len(gs.points), None))
    return MeshDomain(gs.points, gs.squencs, gs.shapes, spacing, nodes, mesh_types)


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


    def __str__(self):
        return 'Face 1: ' + str(self.face1) + '\nFace 2: ' + str(self.face2)


    def flip(self):
        face1, face2 = self.face1, self.face2
        self.face1, self.face2 = face2, face1


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


def face_merge(face_pair_list: list, minsim_1: float = 0, minsim_2: float = 0, minsim_all: float = 0) -> list:
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
    
    while restart:
        restart = False
        ll = len(face_pair_list)

        for i in range(ll):
            fpi1, fpi2 = face_pair_list[i].face1, face_pair_list[i].face2
            for j in range(i+1,ll):
                fpj1, fpj2 = face_pair_list[j].face1, face_pair_list[j].face2
                ls1 = len(set(fpj1 + fpi1))
                ls2 = len(set(fpj2 + fpi2))
                li1, li2 = len(fpi1), len(fpi2)
                lj1, lj2 = len(fpj1), len(fpj2)
                sim_1 = (li1 + lj1 - ls1) / min(li1, lj1)
                sim_2 = (li2 + lj2 - ls2) / min(li2, lj2)
                sim_all = (li2 + lj2 + li1 + lj1 - ls1 - ls2) / (min(li1, lj1) + min(li2, lj2))
                if (sim_1 >= minsim_1) and (sim_2 >= minsim_2) and (sim_all >= minsim_all):
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
    b1 = np.sort([p11[0], p0[0], p21[0]])[1] == p0[0]
    b2 = np.sort([p11[0], p0[0], p21[0]])[1] == p0[0]
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
        for j in range(raylen-1, i, -1):
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
    while True:
        b1, b2n = blist1[bi], blist2[bi]
        b2 = b2n - 1                                # just need it to be different than b2n to run the first loop
        while b2n != b2:
            b2 = b2n
            b2n = np.max(blist2[np.nonzero(blist1 <= b2)[0]])
        blist1_n.append(b1), blist2_n.append(b2)
        if np.any(blist1 > b2):
            bi = np.nonzero(blist1 > b2)[0][0]
        else:
            break
    # Generate curve indexes
    cavities = []
    for i in range(len(blist1_n)):
        cavities.append(list(range(blist1_n[i], blist2_n[i] + 1)))

    return cavities


# TESTING SECTIONS

# def preview_mesh(dm: MeshDomain, div11 = [], div12 = [], div21 = [], div22 = []):
#     """
#     Meshing algorithm.

#     Args:
#         dm: MeshDomain. Each pair of sides must have equal amount of points.

#     Returns:
#         list containing all the subdomains.

#     """
#     reps = 10
#     n, m = len(div11), len(div21)
#     # Calculate the division fractions for every division line. The division fractions transition linearly from one edge to another.
#     div11 = np.repeat(div11[:,np.newaxis], m, axis=0)
#     div12 = np.repeat(div12[:,np.newaxis], m, axis=0)
#     div21 = np.transpose(np.repeat(div21[:,np.newaxis], n, axis=0))
#     div22 = np.transpose(np.repeat(div22[:,np.newaxis], n, axis=0))
#     div1 = ((div12 - div11) * div22 - div12) / ((div12 - div11) * (div22 - div21) - 1)
#     div2 = ((div22 - div21) * div12 - div22) / ((div12 - div11) * (div22 - div21) - 1)
#     div2 = np.transpose(div2)
#     # Calculate boundarie conditions
#     b11 = gmt.crv_div(dm.s11, div11[0])
#     b12 = gmt.crv_div(dm.s12, div12[0])
#     b21 = gmt.crv_div(dm.s21, div21[:,0])
#     b22 = gmt.crv_div(dm.s22, div12[:,0])
#     # Calculate start conditions
#     # Converge upon the grid
#     for _ in range(reps):
#         for segi in range(n):
#             pass
