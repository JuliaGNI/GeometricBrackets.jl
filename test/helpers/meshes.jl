using SimpleSplines: UniformMesh, GradedMesh, RandomMesh

# The three mesh families that the spline-space tests run over, by name and constructor.
const SPLINE_MESHES = ((:uniform, n -> UniformMesh(n, 2π)),
    (:graded, n -> GradedMesh(n, 2π)),
    (:random, n -> RandomMesh(n, 2π)))
