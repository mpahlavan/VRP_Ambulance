#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate LaTeX tables for Ambulance Routing results.
Focuses on deterministic problem variants only.
"""

import torch
import os
from argparse import ArgumentParser

# LaTeX formatting templates
REF_CELL_STR = r"${}{{{:5.2f} \pm {:5.2f}}}$"
CELL_STR = r"${}{{{:5.2f} \pm {:5.2f} ({:3.0f}\%)}}$"

def get_mean_std_gap(sample, ref=None, outlier_factor=1.5):
    """Calculate mean, std, and gap with outlier removal"""
    if len(sample) == 0:
        return float('nan'), float('nan'), float('nan')
    
    q1, _ = sample.kthvalue(max(1, sample.numel()//4))
    q3, _ = sample.kthvalue(max(1, 3*sample.numel()//4))
    mask_outlier = (q1 - outlier_factor*(q3-q1) <= sample) \
            & (sample <= q3 + outlier_factor*(q3-q1))
    
    if mask_outlier.sum() == 0:
        masked = sample
    else:
        masked = sample[mask_outlier]
    
    gap = None if ref is None else 100*(masked / ref[mask_outlier] - 1).mean()
    return masked.mean(), masked.std(), gap

def parse_args():
    parser = ArgumentParser(description="Generate LaTeX tables for PVRP results")
    parser.add_argument("--results-dir", type=str, default="./results",
                       help="Directory containing result files")
    parser.add_argument("--output-file", type=str, default="pvrp_results.tex",
                       help="Output LaTeX file")
    parser.add_argument("--problem-sizes", nargs='+', type=int, 
                       default=[10, 20, 50], help="Problem sizes to analyze")
    # Manual date specification for OR-Tools and learned results
    parser.add_argument("--ortools-date", type=str, default="250727-2125",
                       help="Date folder for OR-Tools results")
    parser.add_argument("--learned-date", type=str, default="250727-2126", 
                       help="Date folder for learned model results")
    return parser.parse_args()

def find_result_files(results_dir, problem_size, ortools_date, learned_date):
    """
    Find result files for OR-Tools and learned model based on manual date specification
    """
    n, m = problem_size, max(2, problem_size // 5)
    
    # OR-Tools results path
    ortools_dir = f"pvrp_n{n}m{m}_{ortools_date}"
    ortools_path = os.path.join(results_dir, ortools_dir, "ort.pyth")
    
    # Learned model results paths
    learned_dir = f"pvrp_n{n}m{m}_{learned_date}"
    learned_base_path = os.path.join(results_dir, learned_dir)
    greedy_path = os.path.join(learned_base_path, "mardan_greedy.pyth")
    sample_path = os.path.join(learned_base_path, "mardan_sample100.pyth")
    
    return {
        'ortools': ortools_path if os.path.exists(ortools_path) else None,
        'greedy': greedy_path if os.path.exists(greedy_path) else None,
        'sample': sample_path if os.path.exists(sample_path) else None,
        'ortools_dir': ortools_dir,
        'learned_dir': learned_dir
    }

def generate_main_comparison_table(args):
    """Generate main comparison table for PVRP"""
    print(r"""\begin{table}[htbp]
\centering
\caption{PVRP (Ambulance Routing) Performance Comparison: OR-Tools vs. Learned Model}
\label{tab:pvrp_main_results}
\begin{tabular}{|c|c|c|c|c|}
\hline
\multirow{2}{*}{Problem Size} & \multicolumn{3}{c|}{Method} & Performance \\
\cline{2-4}
& OR-Tools & Learned (Greedy) & Learned (Sample) & Best Gap (\%) \\
\hline""")

    for n in args.problem_sizes:
        m = max(2, n // 5)
        print(f"\\multirow{{1}}{{*}}{{N={n}, M={m}}}")
        
        # Find result files
        files = find_result_files(args.results_dir, n, args.ortools_date, args.learned_date)
        
        try:
            if files['ortools'] and files['greedy']:
                # Load OR-Tools results
                ort_data = torch.load(files['ortools'], map_location='cpu')
                ort_costs = ort_data["costs"]
                
                # Load greedy results
                greedy_costs = torch.load(files['greedy'], map_location='cpu')
                
                # Load sample results (optional)
                sample_costs = None
                if files['sample']:
                    sample_costs = torch.load(files['sample'], map_location='cpu')
                
                # Calculate statistics
                ort_mean, ort_std, _ = get_mean_std_gap(ort_costs)
                greedy_mean, greedy_std, greedy_gap = get_mean_std_gap(greedy_costs, ort_costs)
                
                if sample_costs is not None:
                    sample_mean, sample_std, sample_gap = get_mean_std_gap(sample_costs, ort_costs)
                else:
                    sample_mean = sample_std = sample_gap = float('nan')
                
                # Determine best method
                gaps = [0, greedy_gap, sample_gap]
                gaps_clean = [g for g in gaps if not torch.isnan(torch.tensor(g))]
                best_gap = min(gaps_clean) if gaps_clean else 0
                
                # Format table cells
                ort_cell = REF_CELL_STR.format("\\boldsymbol" if best_gap == 0 else "", ort_mean, ort_std)
                greedy_cell = CELL_STR.format("\\boldsymbol" if greedy_gap == best_gap else "", 
                                            greedy_mean, greedy_std, greedy_gap)
                
                if not torch.isnan(torch.tensor(sample_gap)):
                    sample_cell = CELL_STR.format("\\boldsymbol" if sample_gap == best_gap else "",
                                                sample_mean, sample_std, sample_gap)
                else:
                    sample_cell = "---"
                
                best_gap_display = f"{best_gap:+.1f}" if best_gap != 0 else "0.0"
                
                print(f"& {ort_cell} & {greedy_cell} & {sample_cell} & {best_gap_display} \\\\")
                
            else:
                missing_files = []
                if not files['ortools']:
                    missing_files.append("OR-Tools")
                if not files['greedy']:
                    missing_files.append("Greedy")
                missing_str = ", ".join(missing_files)
                print(f"& \\multicolumn{{4}}{{c|}}{{Missing: {missing_str} (Dirs: {files['ortools_dir']}, {files['learned_dir']})}} \\\\")
            
        except Exception as e:
            print(f"& \\multicolumn{{4}}{{c|}}{{Error loading files: {str(e)[:30]}}} \\\\")
        
        print("\\hline")
    
    print(r"""\end{tabular}
\end{table}""")

def generate_detailed_metrics_table(args):
    """Generate detailed metrics comparison table"""
    print(r"""\begin{table}[htbp]
\centering
\caption{Detailed PVRP Performance Metrics}
\label{tab:pvrp_detailed_metrics}
\begin{tabular}{|c|c|c|c|c|}
\hline
Problem & Method & Cost & Gap & Data Source \\
Size & & (Mean ± Std) & (\%) & (Directory) \\
\hline""")

    for n in args.problem_sizes:
        m = max(2, n // 5)
        
        # Find result files
        files = find_result_files(args.results_dir, n, args.ortools_date, args.learned_date)
        
        print(f"\\multirow{{3}}{{*}}{{N={n}, M={m}}}")
        
        try:
            if files['ortools']:
                ort_data = torch.load(files['ortools'], map_location='cpu')
                ort_costs = ort_data["costs"]
                ort_mean, ort_std, _ = get_mean_std_gap(ort_costs)
                print(f"& OR-Tools & {ort_mean:.1f} ± {ort_std:.1f} & 0.0 & {files['ortools_dir']} \\\\")
            else:
                print(f"& OR-Tools & --- & --- & Missing \\\\")
            
            if files['greedy']:
                greedy_costs = torch.load(files['greedy'], map_location='cpu')
                if files['ortools']:
                    greedy_mean, greedy_std, greedy_gap = get_mean_std_gap(greedy_costs, ort_costs)
                    print(f"& Learned (G) & {greedy_mean:.1f} ± {greedy_std:.1f} & {greedy_gap:+.1f} & {files['learned_dir']} \\\\")
                else:
                    greedy_mean, greedy_std, _ = get_mean_std_gap(greedy_costs)
                    print(f"& Learned (G) & {greedy_mean:.1f} ± {greedy_std:.1f} & --- & {files['learned_dir']} \\\\")
            else:
                print(f"& Learned (G) & --- & --- & Missing \\\\")
            
            if files['sample']:
                sample_costs = torch.load(files['sample'], map_location='cpu')
                if files['ortools']:
                    sample_mean, sample_std, sample_gap = get_mean_std_gap(sample_costs, ort_costs)
                    print(f"& Learned (S) & {sample_mean:.1f} ± {sample_std:.1f} & {sample_gap:+.1f} & {files['learned_dir']} \\\\")
                else:
                    sample_mean, sample_std, _ = get_mean_std_gap(sample_costs)
                    print(f"& Learned (S) & {sample_mean:.1f} ± {sample_std:.1f} & --- & {files['learned_dir']} \\\\")
            else:
                print(f"& Learned (S) & --- & --- & Missing \\\\")
                
        except Exception as e:
            print(f"& Error & \\multicolumn{{3}}{{c|}}{{Failed to load data}} \\\\")
            print(f"& & \\multicolumn{{3}}{{c|}}{{}} \\\\")
            print(f"& & \\multicolumn{{3}}{{c|}}{{}} \\\\")
        
        print("\\hline")

    print(r"""\end{tabular}
\end{table}""")

def generate_ambulance_specific_table(args):
    """Generate ambulance-specific summary table"""
    print(r"""\begin{table}[htbp]
\centering
\caption{Ambulance Routing Performance Summary}
\label{tab:ambulance_summary}
\begin{tabular}{|c|c|c|c|c|}
\hline
Problem & OR-Tools & Learned & Performance & Assessment \\
Size & Available & Available & Gap & \\
\hline""")

    overall_available = 0
    overall_gaps = []
    
    for n in args.problem_sizes:
        m = max(2, n // 5)
        
        files = find_result_files(args.results_dir, n, args.ortools_date, args.learned_date)
        
        ortools_status = "✓" if files['ortools'] else "✗"
        learned_status = "✓" if files['greedy'] else "✗"
        
        if files['ortools'] and files['greedy']:
            try:
                ort_data = torch.load(files['ortools'], map_location='cpu')
                greedy_data = torch.load(files['greedy'], map_location='cpu')
                
                ort_costs = ort_data["costs"]
                _, _, gap = get_mean_std_gap(greedy_data, ort_costs)
                
                gap_str = f"{gap:+.1f}\\%"
                overall_gaps.append(gap)
                overall_available += 1
                
                if gap < -5:
                    assessment = "Excellent"
                elif gap < 0:
                    assessment = "Good"
                elif gap < 10:
                    assessment = "Fair"
                else:
                    assessment = "Poor"
                    
            except:
                gap_str = "Error"
                assessment = "Failed"
        else:
            gap_str = "---"
            assessment = "No Data"
        
        print(f"N={n}, M={m} & {ortools_status} & {learned_status} & {gap_str} & {assessment} \\\\")
        print("\\hline")
    
    # Add summary row
    if overall_gaps:
        avg_gap = sum(overall_gaps) / len(overall_gaps)
        print(f"\\multicolumn{{3}}{{|c|}}{{Overall Average Gap}} & {avg_gap:+.1f}\\% & ")
        if avg_gap < 0:
            print("Good \\\\")
        elif avg_gap < 10:
            print("Acceptable \\\\")
        else:
            print("Needs Improvement \\\\")
        print("\\hline")

    print(r"""\end{tabular}
\end{table}""")

def main():
    args = parse_args()
    
    print("% Ambulance Routing) Results Tables")
    print("% Generated for deterministic problem variants only")
    print("% OR-Tools date:", args.ortools_date)
    print("% Learned model date:", args.learned_date)
    print("% =====================================================\n")
    
    # Generate main comparison table
    print("% Main Performance Comparison")
    generate_main_comparison_table(args)
    print()
    
    # Generate detailed metrics table  
    print("% Detailed Performance Metrics")
    generate_detailed_metrics_table(args)
    print()
    
    # Generate ambulance-specific summary
    print("% Ambulance Routing Summary")
    generate_ambulance_specific_table(args)
    print()
    
    print("% End of PVRP Results Tables")
    print(f"% Files analyzed from directories: *_{args.ortools_date} and *_{args.learned_date}")

if __name__ == "__main__":
    main()