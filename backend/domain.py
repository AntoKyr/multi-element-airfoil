#--------- READ ME----------------------------------------------------------------
# A package containing the functions for the generation of mesh domains.
#---------------------------------------------------------------------------------

import numpy as np
import geometrics as gmt
import functbook as fnb
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
        nodes (list): Contains lists for each sequence, with the following: [nodenum, mesh_type, coef]
        mesh_types (list): Contains strings describing the meshing method for each shape. 'hex' for structured hexa mesh, 'tri' for unstructured tri mesh, 'void' to remove domain from control volume (for example the airfoil body)

    Methods:
        i dont care to list them

    """

    def __init__(self, points: np.ndarray, squencs: list, shapes: list, spacing: list, nodes: list, mesh_types: list):
        super(MeshDomain, self).__init__(points, squencs, shapes)
        self.nodes = nodes
        self.spacing = spacing
        self.mesh_types = mesh_types


    def __str__(self):
        return super(MeshDomain, self).__str__() + '\nSpacing: ' + str(self.spacing) + '\nNodes: ' + str(self.nodes) + '\nMesh Types: ' + str(self.mesh_types)


    def printsizes(self):
        """
        Print the sizes of the MeshDomain attributes. Mainly for debugging.
        """
        print('---MeshDomain Sizes---')
        print('points: ', len(self.points))
        print('spacing: ', len(self.spacing))
        print('squencs: ', len(self.squencs))
        print('nodes: ', len(self.nodes))
        print('shapes: ', len(self.shapes))
        print('mesh_types: ', len(self.mesh_types))
        print('----------------------')


    def remove_point(self, point_i: int) -> list:
        self.spacing.pop(point_i)
        return super(MeshDomain, self).remove_point(point_i)


    def remove_sequence(self, squence_i: int) -> list:
        self.nodes.pop(squence_i)
        return super(MeshDomain, self).remove_sequence(squence_i)


    def clear_duplicates(self, tol: float):
        """
        Clear all duplicate points, sequences and shapes from the GeoShape.

        Args:
            tol: the maximum distance two points must have to be considered the same point.

        """
        # Points
        i = 0 
        while i < len(self.points) - 1:
            j = i + 1
            while j < len(self.points):
                if gmt.comcheck(self.points[i], self.points[j], tol):
                    self.replace_point(j, i)
                    self.remove_point(j)
                else:
                    j += 1
            i += 1
        # Sequences
        squencs = self.squencs
        for squence in squencs:
            i = 0
            while i < len(squence) - 1:
                if squence[i] == squence[i-1]:
                    squence.pop(i)
                else:
                    i += 1

        i = 0
        while i < len(squencs) - 1:
            j = i + 1
            while j < len(squencs):
                if (squencs[i] == squencs[j]) or (squencs[i] == list(np.flip(squencs[j]))):
                    self.replace_sequence(j, i)
                    self.remove_sequence(j)
                else:
                    j += 1
            i += 1
        # Shapes
        shapes = self.shapes
        for shape in shapes:
            i = 0
            while i < len(shape) - 1:
                if shape[i] == shape[i-1]:
                    shape.pop(i)
                else:
                    i += 1

        i = 0
        while i < len(shapes) - 1:
            j = i + 1
            while j < len(shapes):
                if (shapes[i] == shapes[j]) or (shapes[i] == list(np.flip(shapes[j]))):
                    self.shapes.pop(j)
                else:
                    j += 1
            i += 1


    # DOMAIN GENERATION
    def layer_domain(self, squenc_i: int, thickness_func: Callable[[np.ndarray], np.ndarray], mesh_type: str = 'hex', side_nodes: list = None) -> int:
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
        lt = thickness_func(crv)
        lt = np.transpose([lt, lt])
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
        self.spacing = self.spacing + list(np.full(len(crv), None))
        self.nodes = self.nodes + [self.nodes[squenc_i], side_nodes, side_nodes]
        # Return stuff
        return len(self.shapes) - 1


    def unstruct_domain(self, squencs: list) -> int:
        """
        Generate an unstructured domain.
        """
        self.shapes.append(squencs)
        self.mesh_types.append('tri')
        return len(self.shapes) - 1


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


    # DOMAIN MANIPULATION
    def stitch_domains(self, shape_1: int, shape_2: int, nd: int, squenc_1: int = None, squenc_2: int = None, deform_indx: int = 0, boundary_indx: int = 0):
        """
        Attach two "parallel" domains by deforming them as needed and merging their borders.

        Args:
            shape_1: the index of the first shape
            shape_2: the index of the second shape
            nd: if 1 the nodes are determined by the first shape, if 2, by the second
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
        self.spacing = self.spacing + list(np.full(len(crvm), None)) 
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

        # Determine nodes according to node dominance
        if nd == 1:
            new_nodes = self.nodes[seqis_1[i1]]
        elif nd == 2:
            new_nodes = self.nodes[seqis_2[i2]]
            if flipindx == -1:
                new_nodes = np.flip(new_nodes)
            
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


    def hexdomfix(self, shape_i: int, dom: str = 'max'):
        """
        Fix node imbalances of a hex domain, by equating the number of nodes on opposite sides.

        Args:
            shape_i: the index of the shape to be fixed
            dom: if 'max': the max no nodes of the side pairs is used, elif 'mean': the mean no nodes is used, elif 'min': the minimum no nodes is used

        """
        sqis = self.shapes[shape_i]
        order = self.shape_clarify(shape_i)[0]
        if dom == 'max':
            func = np.max
        elif dom == 'mean':
            func = lambda x: round(np.mean(x))
        elif dom == 'min':
            func = np.min
        
        non0 = self.nodes[sqis[order[0]]][0]
        non1 = self.nodes[sqis[order[1]]][0]
        non2 = self.nodes[sqis[order[2]]][0]
        non3 = self.nodes[sqis[order[3]]][0]
        self.nodes[sqis[order[0]]][0] = func([non0, non2])
        self.nodes[sqis[order[2]]][0] = func([non0, non2])
        self.nodes[sqis[order[1]]][0] = func([non1, non3])
        self.nodes[sqis[order[3]]][0] = func([non1, non3])


    def simple_boundary_layer_gen(self, maxblt: float, vd: float, bltsc: float = 0.95, blcoef: float = 1.2) -> list:
        """
        Generate simple boundary layer mesh domains for auto-generated airfoil sections.

        Args:
            maxblt: maximum boundary layer thickness
            vd: vertical density of the boundary layer
            bltsc: boundary layer thickness smoothening coef
            blcoef: boundary layer meshing coef
        
        Returns:
            outer shells

        """
        points = self.points
        squencs = self.squencs
        shapes = self.shapes

        outer_shell = []
        # generate boundary layers
        for shi in range(len(shapes)):
            # construct polygon and measure angles
            poli = squencs[shapes[shi][0]][0:-1]
            vertexangs = []
            for i in range(1, len(shapes[shi])):
                indxs = list(squencs[shapes[shi][i]])
                vertexangs.append(gmt.vectorangle(points[indxs[1]] - points[indxs[0]], points[poli[-1]] - points[indxs[0]]))
                poli = poli + indxs[0:-1]

            # get boundary layer thickness
            if len(self.shapes) > 1:
                lp = points[poli]
                nlp = np.delete(points, poli, axis=0)
                dists = np.min(gmt.crv_dist(lp, nlp), axis=1)/3
                maxblta = np.full(len(dists), maxblt)
                blt = np.min([dists, maxblta], axis=0)

            # check for cavities
            blti1 = 0
            for sqi in shapes[shi]:
                indxs = np.array(squencs[sqi])
                cavlist = cavity_id(points[indxs], maxblt)
                blti2 = blti1 + len(indxs)
                blti = np.arange(blti1, blti2)
                for cav in cavlist:
                    cavblt = - 1/min(gmt.crv_curvature(points[indxs[cav]]))
                    blt[blti[cav]] = np.min([np.full(len(cav), cavblt), blt[blti[cav]]], axis=0)
                blti1 = blti2 - 1

            # smoothen 
            blt = smoothen_blt(lp, blt, bltsc)

            # measure angles
            vertexangs = np.array(vertexangs)
            negi = np.nonzero(vertexangs < 0)[0]
            if len(negi) > 0:
                vertexangs[negi] = 2 * np.pi + vertexangs[negi]
            reflexi = np.nonzero(vertexangs < np.pi/2)[0]
            acutei = np.nonzero(vertexangs > 3*np.pi/2)[0]

            # build domains and patch them
            sqios1 = len(self.squencs)   # surprise tool
            # first domain
            nn = int(vd * maxblt)
            blti1 = 0
            blti2 = blti1 + len(squencs[shapes[shi][0]])
            blti = np.arange(blti1, blti2)
            self.layer_domain(shapes[shi][0], lambda x: blt[blti], side_nodes=[nn,'Progression', blcoef])
            blti1 = blti2 - 1
            lsh = len(shapes[shi])
            i = 1
            # repeat
            while i < lsh:
                blti2 = blti1 + len(squencs[shapes[shi][i]])
                blti = np.arange(blti1, blti2)
                if i == lsh - 1:
                    blti[-1] = 0

                if (i-1) in reflexi:
                    self.reflex_domain(shapes[shi][i], shapes[shi][i+1])
                    blti1 += len(shapes[shi][i]) + len(shapes[shi][i]) - 2
                    i += 2

                elif (i-1) in acutei:
                    self.layer_domain(shapes[shi][i], lambda x: blt[blti], side_nodes=[nn,'Progression', blcoef])
                    blti1 = blti2 - 1
                    i += 1
                
                else:
                    self.layer_domain(shapes[shi][i], lambda x: blt[blti], side_nodes=[nn,'Progression', blcoef])
                    self.stitch_domains(len(self.shapes)-1, len(self.shapes)-2, nd=2)
                    blti1 = blti2 - 1
                    i += 1
            
            # get outer shell
            sqios2 = len(self.squencs)
            os = self.outer_shell(list(range(sqios1, sqios2)))
            outer_shell = outer_shell + os
        return outer_shell


    def simple_trailpoint_gen(self, hfl: float, hft: float, xd: float, ss: float, es: float, nn: int, sppo: float):
        """
        Generate trail points and spacings.

        Args:
            md: MeshDomain
            hfl: leading edge trail height factor
            hft: trailing edge trail height factor
            xd: x distance
            ss: min spacing at the start of the trail
            es: max spacing at the end of the trail
            nn: number of points
            sppo: spacing power, influences the spacing distribution of the trail points

        """
        # get all shapes with positive void coeffiecient
        pvci = np.nonzero(np.array(self.mesh_types) == 'void')[0]
        vdshapes = []
        for i in pvci:
            vdshapes.append(self.shapes[i])

        for shape in vdshapes:
            # Get trail vectors and points
            tpi0 = self.squencs[shape[0]][0]
            tpi1 = self.squencs[shape[0]][1]
            tpi2 = self.squencs[shape[-1]][-2]
            v1 = self.points[tpi1] - self.points[tpi0]
            v2 = self.points[tpi2] - self.points[tpi0]
            tvec = - gmt.bisector_vct(v1, v2)
            p1 = self.points[tpi0]
            # get trail height
            th1 = fnb.gen_polysin_trail_height(hft)(gmt.vectorangle(tvec))
            # get all points
            p2 = [xd, th1 + p1[1]]
            lf11 = [0, th1 + p1[1]]
            lf12 = gmt.lfp(p1, p1+tvec)
            p0 = gmt.lnr_inters(lf11, lf12)
            # build trail array
            points = gmt.bezier([p1, p0, p2])(np.linspace(0,1,nn+1)[1:])
            # calculate spacing
            spacings = list(np.linspace(ss**(1/sppo), es**(1/sppo), nn)**sppo)
            # save
            self.points = np.vstack((self.points, points))
            self.spacing = self.spacing + spacings

            if abs(gmt.vectorangle(tvec)) > np.pi/6:
                # get le
                tp = self.points[tpi0]
                pi = []
                for sqi in shape:
                    pi += self.squencs[sqi][0:-1]
                p2i = pi[np.argmax(np.hypot(self.points[pi,0] - tp[0], self.points[pi,1] - tp[1]))]
                p1 = self.points[p2i]
                # get trail height
                th2 = fnb.gen_polysin_trail_height(hfl)(gmt.vectorangle(tvec))
                # get all points
                p2 = [xd, th2 + p1[1]]
                lf21 = [0, th2 + p1[1]]
                lf22 = gmt.lfp(p1, p1+tvec)
                p0 = gmt.lnr_inters(lf21, lf22)
                # build trail array
                points = gmt.bezier([p1, p0, p2])(np.linspace(0,1,nn+1)[1:])
                # save
                self.points = np.vstack((self.points, points))
                self.spacing = self.spacing + spacings


    def simple_controlvol_gen(self, outer_shells: list, xd: float, h: float, bs: float):
        """
        Generate a simple control volume. Meant to be used after trail point gen.

        Args:
            xd: x distance
            h: height
            bs: border spacing

        Returns:
            indexes of the control volume sequences

        """
        # new points and spacings
        points = [[xd,h],[-h,h],[-h,-h],[xd,-h]]
        spacings = [bs, bs, bs, bs]
        i1 = len(self.points)
        self.points = np.vstack((self.points, points))
        self.spacing = self.spacing + spacings
        # get all points at cv border
        cvbi = np.nonzero(self.points[:,0] >= 0.999*xd)[0]
        # sort them by y value
        cvbi = cvbi[np.argsort(self.points[cvbi,1])]
        # generate sequences
        trailseq = []
        for i in range(len(cvbi)-1):
            trailseq.append([cvbi[i], cvbi[i+1]])
        seqs = trailseq + [[i1, i1+1], [i1+1, i1+2], [i1+2, i1+3]]
        i1 = len(self.squencs)
        self.squencs = self.squencs + seqs
        i2 = len(self.squencs)
        self.nodes = self.nodes + list(np.full(i2-i1, None))
        # Generate da big domain
        self.shapes.append(list(range(i1,i2)) + outer_shells)
        self.mesh_types.append('tri')


    def simple_prox_layer(self, proxd: float, sp: float):
        """
        Generate proximity layer points and spacings.

        Args:
            proxd: the distance of the proximity layer
            sp: spacing of proximity layer
        
        """
        clf = 0.95
        vdshapes = np.nonzero(np.array(self.mesh_types)=='void')[0]
        proxp = np.array([], dtype=float).reshape(0,2)
        shapp = np.array([], dtype=float).reshape(0,2)
        for i in vdshapes:
            shape = self.shapes[i]
            pi = []
            for sqi in shape:
                pi = pi + self.squencs[sqi][0:-1]
            # calc prox layer
            shapp = np.vstack((shapp, self.points[pi]))
            proxp = np.vstack((proxp, self.points[pi] + gmt.parallcrv(self.points[pi]) * proxd))
        
        # remove all points closer to the shapes than they should be
        dists = gmt.crv_dist(proxp, shapp)
        mindists = np.min(dists, axis=1)
        vali = np.nonzero(mindists >= clf * proxd)[0]
        proxp = proxp[vali]
        # add points and spacings
        spac = list(np.full(len(proxp), sp))
        self.points = np.vstack((self.points, proxp))
        self.spacing = self.spacing + spac


def element_sort(gs: gmt.GeoShape) -> gmt.GeoShape:
    """
    Sort the shapes of a GeoShape repressenting a multi element airfoil, from trailing to leading edge.

    Args:
        gs: GeoShape to be sorted
    
    Returns:
        Sorted GeoShape

    """
    shapes = gs.shapes
    squencs = gs.squencs
    points = gs.points
    # Get all leading and trailing edges
    lel = []
    tel = []
    for shape in shapes:
        te = points[squencs[shape[0]][0]]
        # Get all points of shape
        shapoints = points[np.concatenate(list(squencs[i] for i in shape))]
        le = shapoints[np.argmax(np.hypot(shapoints[:,0] - te[0], shapoints[:,1] - te[1]))]
        tel.append(te)
        lel.append(le)

    tel, lel = np.array(tel), np.array(lel)
    dists = gmt.crv_dist(tel, lel)
    lte = tel[np.argmax(np.max(dists, axis=1))]
    charvec = np.sum(lel - tel, axis=0)
    sortvals = (lel - np.repeat(lte[np.newaxis,:], len(lel), axis=0)) @ charvec
    sortindxs = np.argsort(sortvals)
    shapes = list(shapes[i] for i in sortindxs)
    return gmt.GeoShape(points, squencs, shapes)


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
    points = np.array(gs.points)
    squencs = list(gs.squencs)
    shapes = list(gs.shapes)
    return MeshDomain(points, squencs, shapes, spacing, nodes, mesh_types)


def smoothen_blt(c: np.ndarray, t: np.ndarray, sf: float) -> np.ndarray:
    """
    Smoothen boundary layer thickness transitions.

    Args:
        c: curve
        t: thickness
        sf: smoothening factor (0 < sf < 1), the greater the factor, the smoother the thickness transtions, at 1, thickness is the same everywhere and equal to minimum.
    
    Returns:
        smoothened thickness

    """
    crvlen = gmt.crv_len(c)
    maxdt = (crvlen[1:] - crvlen[0:-1]) * (1-sf)/2
    dt = t[1:] - t[0:-1]
    while np.any(np.abs(dt) > maxdt):
        pdti = np.nonzero(dt > maxdt)[0]
        ndti = np.nonzero(-dt > maxdt)[0]
        t[pdti + 1] = t[pdti] + 0.99 * maxdt[pdti]
        t[ndti] = t[ndti + 1] + 0.99 * maxdt[ndti]
        dt = t[1:] - t[0:-1]
    return t


def simple_section_prep(gs: gmt.GeoShape, dens: float, dlp: float, coef: float, method = 'Bump') -> MeshDomain:
    """
    Prepare an auto-generated airfoil section to be meshed, by making it into a MeshDomain and adding nodes.

    Args:
        dens: the number of nodes per unit length
        dlp: density length - power
        coef: the coefficient of the nodes

    """
    points = gs.points
    squencs = gs.squencs
    shapes = gs.shapes

    for shape in shapes:
        tep = points[squencs[shape[0]][0]]
        chordlen = 0
        endlen = 0
        for sqi in shape:
            vlen = np.hypot(tep[0] - points[squencs[sqi],0], tep[1] - points[squencs[sqi],1])
            endlen = max(endlen, np.linalg.norm(points[squencs[sqi][-1]] - tep))
            maxi = np.argmax(vlen)
            if vlen[maxi] > chordlen:
                lesqi = sqi
                lei = maxi
                chordlen = vlen[maxi]
        
        if endlen < 0.8 * chordlen:
            gs.split_sequence(lesqi, [lei])

    md = gs2md(gs)

    for i in range(len(md.squencs)):
        crvlen = gmt.crv_len(points[md.squencs[i]])[-1]
        md.nodes[i] = [int(dens * crvlen**dlp), method, coef]
    
    md.mesh_types = ['void'] * len(md.mesh_types)
    return md


# MORPHOLOGY
def cavity_id(c: Union[list,tuple,np.ndarray], char_len: float) -> list:
    """
    Identify cavities of a curve.

    Args:
        c: [[x0, y0], [x1, y1], ... , [xn, yn]] the matrix containing all the point coordinates of the curve
        char_len: characteristic length, the smaller it is, the more curved, cavities must be to be identified.
    
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
    if len(blist1) > 0:
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
