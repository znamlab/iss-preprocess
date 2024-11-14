"""We had some issues with circular dependencies in the past, so make a graph
of relative imports (using import_depts)."""

import pathlib

import graphviz
from import_deps import ModuleSet

# First initialise a ModuleSet instance with a list str of modules to track
pkg_paths = pathlib.Path("iss_preprocess").glob("**/*.py")
module_set = ModuleSet([str(p) for p in pkg_paths])

dot = graphviz.Digraph(comment="Relative imports graph of iss_preprocess")

# first create all the nodes
for mod in sorted(module_set.by_name.keys()):
    dot.node(mod)

# then create all the edges
for mod in sorted(module_set.by_name.keys()):
    for dep in module_set.mod_imports(mod):
        dot.edge(dep, mod)

dot.render("docs/dep_graph", format="png", cleanup=True)
