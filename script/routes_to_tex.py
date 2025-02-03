# routes_to_tex.py
from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.externals import ort_solve
from marpdan.utils import load_old_weights
import torch
import subprocess
import os

TIKZ_TMPL = r"""\documentclass[tikz]{{standalone}}
\begin{{document}}
\begin{{tikzpicture}}[x=0.1mm, y=0.1mm,
    depot/.style={{draw, star, fill=yellow!40}},
    cust/.style={{draw, circle, fill=white}}]
    
    \node[depot] (0) at ({},{}) {{0}};
    {}
    
    {}
\end{{tikzpicture}}
\end{{document}}"""

def generate_tikz(nodes, routes, loc_scale):
    depot = nodes[0,:2]*loc_scale
    customers = nodes[1:,:2]*loc_scale
    spoilage = nodes[1:,3]*loc_scale
    
    cust_nodes = "\n".join(
        f"\\node[cust] ({i}) at ({x},{y}) {{{i}}};"
        f"\\node[font=\\tiny] at ({x},{y-15}) {{{spoilage:.0f}}};"
        for i, (x,y,spoilage) in enumerate(zip(customers[:,0], customers[:,1], spoilage), 1)
    )
    
    routes_tikz = []
    for vid, route in enumerate(routes):
        if not route: continue
        path = [0] + route + [0]
        routes_tikz.append(
            "\\draw[thick] " + " -- ".join(f"(n{p})" for p in path) + ";"
        )
    
    return TIKZ_TMPL.format(depot[0], depot[1], cust_nodes, "\n".join(routes_tikz))

def main():
    args = parse_args()  # Implement proper argument parsing
    data = PVRP_Dataset.generate(1, args.customers_count, args.vehicles_count)
    
    # OR-Tools solution
    ort_routes = ort_solve(data)[0]
    with open("ort_routes.tex", "w") as f:
        f.write(generate_tikz(data.nodes[0], ort_routes, data.nodes[:,:2].max()))
    
    # Learned solution
    learner = AttentionLearner.load(args.model_path)
    _, _, routes = learner(PVRP_Environment(data))
    with open("learned_routes.tex", "w") as f:
        f.write(generate_tikz(data.nodes[0], routes, data.nodes[:,:2].max()))
    
    subprocess.run(["pdflatex", "ort_routes.tex"])
    subprocess.run(["pdflatex", "learned_routes.tex"])

if __name__ == "__main__":
    main()