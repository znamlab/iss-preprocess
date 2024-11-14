"""We had some issues with circular dependencies in the past, so make a graph
of relative imports (using import_depts)."""

import pathlib

import graphviz
from import_deps import ModuleSet
from matplotlib import cm

# First initialise a ModuleSet instance with a list str of modules to track
pkg_paths = pathlib.Path("iss_preprocess").glob("**/*.py")
module_set = ModuleSet([str(p) for p in pkg_paths])

dot = graphviz.Digraph(comment="Relative imports graph of iss_preprocess")
dot.node_attr.update(style="filled", shape="box")

# first create all the nodes
colors = list(cm.get_cmap("Set3").colors)
colors = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in colors]
color_dict = {}
for mod in sorted(module_set.by_name.keys()):
    submod = mod.split(".")[1]
    if submod not in color_dict:
        color_dict[submod] = colors.pop(0)
    dot.node(mod, fillcolor=color_dict[submod])

# then create all the edges
for mod in sorted(module_set.by_name.keys()):
    for dep in module_set.mod_imports(mod):
        dot.edge(dep, mod)

dot.render("docs/dep_graph", format="png", cleanup=True)
