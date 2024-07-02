from sympy import symbols, diff, cos, sqrt, acos, Function

x, y, z, x0, y0, z0 = symbols('x y z x0 y0 z0')
u = symbols('u', cls=Function)

r = sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
phi = acos(z / r)

u = Function('u')(x, y, z)
d2u_dphi2 = diff(u, phi)
print(d2u_dphi2)

if False:
    from mpi4py import MPI
    import gmsh
    from dolfinx.io import gmshio
    import dolfinx.mesh as msh
    import numpy as np
    from dolfinx import plot
    from dolfinx.fem import FunctionSpace, form
    from ufl import Measure, TrialFunction, TestFunction, grad, inner, SpatialCoordinate

    gmsh.initialize()
    comm = MPI.COMM_WORLD
    model_rank = 0
    model = gmsh.model()
    gmsh.model.add("test")

    side_box = 1
    lc = 1e-1
    # Definition of the points
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(side_box, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(side_box, side_box, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, side_box, 0, lc)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    cl1 = [l1, l2, l3, l4]
    s1 = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(cl1)])
    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [l1], tag=1)
    gmsh.model.addPhysicalGroup(2, [s1], tag=1)
    gmsh.model.mesh.generate(2)
    final_mesh, cell_tags, facet_tags = gmshio.model_to_mesh(model, comm, model_rank)
    gmsh.finalize()
        
    tdim = final_mesh.topology.dim
    fdim = tdim - 1

    submesh, entity_map = msh.create_submesh(final_mesh, fdim, facet_tags.find(1))[0:2]
    extract = np.array([entity_map.tolist().index(entity) if entity in entity_map else -1 for entity in range(mesh_num_facets)])
    entity_maps_mesh = {submesh: extract}

    deg    = 2
    family = "Lagrange"

    P1 = element(family, final_mesh.basix_cell(), deg)
    P = functionspace(mesh, P1)

    Q1 = element(family, submesh.basix_cell(), deg-1)
    Q = functionspace(submesh, Q1)

    p, q = TrialFunction(P), TrialFunction(Q)
    v, u = TestFunction(P), TestFunction(Q)

    dx   = Measure("dx", domain=final_mesh, subdomain_data=cell_tags)
    ds   = Measure("ds", domain=final_mesh, subdomain_data=facet_tags)
    dx1  = Measure("dx", domain=submesh)

    x = SpatialCoordinate(submesh)

    e00  = inner(p, v)*dx
    e01  = inner(q, v)*ds(1)

    e10  = inner(1/x[0]*p, u)*dx1



    a    = form(e0x)

