using Aqua
using GeometricBrackets
using Test

# Package-level quality assurance: type piracy, method ambiguities, stale and duplicated
# dependencies, undefined exports, unbound type parameters, `Project.toml` validity. These are
# the faults the rest of the suite is structurally unable to see -- it exercises behaviour, and
# every one of these is a property of the package as a whole.
#
# Piracy is the one this package is most exposed to. Its assembly interface extends SimpleSplines
# generics rather than defining its own, which is what keeps `using` both packages unambiguous,
# and every such method is piracy unless it dispatches on a type defined here. Aqua is what holds
# that line as methods are added to the shared generics.
Aqua.test_all(GeometricBrackets)
