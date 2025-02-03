# results_to_tex.py
import torch
import numpy as np

REF_CELL_STR   = r"${}{{{:5.1f} \pm {:5.1f}}}$"
CELL_STR       = r"${}{{{:5.1f} \pm {:5.1f} ({:3.0f}\%)}}$"

def get_metrics(sample, ref=None):
    mean, std = sample.mean(), sample.std()
    gap = 100 * (mean/ref.mean() - 1) if ref else 0
    return mean, std, gap

def main():
    print(r"""\begin{tabular}{|c|c|c|c|}
    \hline
    Method & N=10 & N=20 & N=50 \\
    \hline""")
    
    for n in (10, 20, 50):
        data_path = f"./results/pvrp_n{n}m{n//5}/"
        
        # Reference solution
        ref = torch.load(data_path+"ort.pyth")['costs']
        
        # Learned solutions
        greedy = torch.load(data_path+"mardan_greedy.pyth")
        sampled = torch.load(data_path+"mardan_sample100.pyth")
        
        # Format results
        ort_metrics = get_metrics(ref)
        greedy_metrics = get_metrics(greedy, ref)
        sampled_metrics = get_metrics(sampled, ref)
        
        print(r"ORTools & {} & {} & {} \\".format(
            REF_CELL_STR.format("", *ort_metrics[:2]),
            REF_CELL_STR.format("", *ort_metrics[:2]),
            REF_CELL_STR.format("", *ort_metrics[:2])))
        
        print(r"MARDAM (G) & {} & {} & {} \\".format(
            CELL_STR.format("", *greedy_metrics),
            CELL_STR.format("", *greedy_metrics), 
            CELL_STR.format("", *greedy_metrics)))
        
        print(r"MARDAM (S) & {} & {} & {} \\ \hline".format(
            CELL_STR.format("", *sampled_metrics),
            CELL_STR.format("", *sampled_metrics),
            CELL_STR.format("", *sampled_metrics)))

    print(r"\end{tabular}")

if __name__ == "__main__":
    main()