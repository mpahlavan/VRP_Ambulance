#!/usr/bin/env python3
"""
PlotNeuralNet visualization for PVRP Actor-Critic Architecture - Version 7 (Final)
Ambulance Routing Problem with Survival Time Constraints

Changes from v6:
- Changed "Patients" to "Input (Patients)"
- Changed "Ambulances" to "Input (Ambulances)"
"""

import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *


arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # ========================================================================
    # TITLE
    # ========================================================================
    r"""\node[font=\Large\bfseries] at (14, 9) {Attention-based Actor-Critic for Ambulance Routing};""",
    r"""\node[font=\normalsize, text=gray] at (14, 8) {Deep Reinforcement Learning with Survival Time Constraints};""",

    # ========================================================================
    # ROW 1: PATIENT PROCESSING (TOP)
    # ========================================================================

    # Patient Input - with spaced dimensions
    to_Conv("input_patient", s_filer="", n_filer="", offset="(0,0,0)", to="(0,0,0)",
            width=2, height=35, depth=35, caption=" "),
    r"""\node[below=0.5cm, font=\small\bfseries] at (input_patient-south) {Input (Patients)};""",
    r"""\node[below=1.0cm, font=\scriptsize] at (input_patient-south) {$[B, N, 4]$};""",
    # Manual dimension labels with proper spacing
    r"""\node[font=\scriptsize, rotate=90, anchor=south] at ([xshift=-0.3cm]input_patient-west) {$N$};""",
    r"""\node[font=\scriptsize, anchor=south west] at ([yshift=0.3cm]input_patient-near) {$4$};""",

    # Depot Input - positioned above patients
    to_Conv("input_depot", s_filer="", n_filer="", offset="(0,5,0)", to="(0,0,0)",
            width=2, height=10, depth=10, caption=" "),
    r"""\node[above=0.3cm, font=\small\bfseries] at (input_depot-north) {Input (Depot)};""",
    r"""\node[font=\scriptsize, rotate=90, anchor=south] at ([xshift=-0.3cm]input_depot-west) {$1$};""",
    r"""\node[font=\scriptsize, anchor=south west] at ([yshift=0.3cm]input_depot-near) {$4$};""",

    # Embedding - with spaced dimensions
    to_Conv("embed", s_filer="", n_filer="", offset="(3.5,0,0)", to="(input_patient-east)",
            height=33, depth=33, width=2.5, caption=" "),
    r"""\node[below=0.5cm, font=\small\bfseries] at (embed-south) {Embedding};""",
    r"""\node[below=1.0cm, font=\scriptsize] at (embed-south) {$4 \rightarrow 128$};""",
    r"""\node[font=\scriptsize, rotate=90, anchor=south] at ([xshift=-0.3cm]embed-west) {$N$};""",
    r"""\node[font=\scriptsize, anchor=south west] at ([yshift=0.3cm]embed-near) {$128$};""",
    to_connection("input_patient", "embed"),

    # Depot Embedding - positioned above patient embedding
    to_Conv("depot_emb", s_filer="", n_filer="", offset="(3.5,0,0)", to="(input_depot-east)",
            height=10, depth=10, width=2.5, caption=" "),
    r"""\node[above=0.3cm, font=\small\bfseries] at (depot_emb-north) {Depot Emb};""",
    r"""\node[font=\scriptsize, rotate=90, anchor=south] at ([xshift=-0.3cm]depot_emb-west) {$1$};""",
    r"""\node[font=\scriptsize, anchor=south west] at ([yshift=0.3cm]depot_emb-near) {$128$};""",
    to_connection("input_depot", "depot_emb"),

    # Concat block - joins depot and patient embeddings
    to_Conv("concat", s_filer="", n_filer="", offset="(2.5,0,0)", to="(embed-east)",
            height=35, depth=35, width=2, caption=" "),
    r"""\node[below=0.5cm, font=\small\bfseries] at (concat-south) {Concat};""",
    r"""\node[font=\scriptsize, rotate=90, anchor=south] at ([xshift=-0.3cm]concat-west) {$N{+}1$};""",
    r"""\node[font=\scriptsize, anchor=south west] at ([yshift=0.3cm]concat-near) {$128$};""",
    to_connection("embed", "concat"),
    to_connection("depot_emb", "concat"),

    # ========================================================================
    # TRANSFORMER ENCODER - Like v2 with MHA + FFN visible
    # ========================================================================

    # Title for all encoders
    r"""\node[font=\normalsize\bfseries, text=blue!70!black] at (13.5, 5.5) {Transformer Encoder (3 Layers $\times$ 8 Heads)};""",

    # Encoder Layer 1 - MHA block
    to_Conv("enc1_mha", s_filer="", n_filer="", offset="(2.5,0,0)", to="(concat-east)",
            height=31, depth=31, width=1.5, caption=" "),
    # Encoder Layer 1 - FFN block
    to_Conv("enc1_ffn", s_filer="", n_filer="", offset="(0.4,0,0)", to="(enc1_mha-east)",
            height=31, depth=31, width=2, caption=" "),
    r"""\node[below=0.5cm, font=\small\bfseries] at (enc1_ffn-south) {Encoder L1};""",
    r"""\node[above=0.2cm, font=\tiny] at (enc1_mha-north) {MHA};""",
    r"""\node[above=0.2cm, font=\tiny] at (enc1_ffn-north) {FFN};""",
    to_connection("concat", "enc1_mha"),

    # Encoder Layer 2 - MHA block
    to_Conv("enc2_mha", s_filer="", n_filer="", offset="(2.5,0,0)", to="(enc1_ffn-east)",
            height=29, depth=29, width=1.5, caption=" "),
    # Encoder Layer 2 - FFN block
    to_Conv("enc2_ffn", s_filer="", n_filer="", offset="(0.4,0,0)", to="(enc2_mha-east)",
            height=29, depth=29, width=2, caption=" "),
    r"""\node[below=0.5cm, font=\small\bfseries] at (enc2_ffn-south) {Encoder L2};""",
    r"""\node[above=0.2cm, font=\tiny] at (enc2_mha-north) {MHA};""",
    r"""\node[above=0.2cm, font=\tiny] at (enc2_ffn-north) {FFN};""",
    to_connection("enc1_ffn", "enc2_mha"),

    # Encoder Layer 3 - MHA block
    to_Conv("enc3_mha", s_filer="", n_filer="", offset="(2.5,0,0)", to="(enc2_ffn-east)",
            height=27, depth=27, width=1.5, caption=" "),
    # Encoder Layer 3 - FFN block
    to_Conv("enc3_ffn", s_filer="", n_filer="", offset="(0.4,0,0)", to="(enc3_mha-east)",
            height=27, depth=27, width=2, caption=" "),
    r"""\node[below=0.5cm, font=\small\bfseries] at (enc3_ffn-south) {Encoder L3};""",
    r"""\node[above=0.2cm, font=\tiny] at (enc3_mha-north) {MHA};""",
    r"""\node[above=0.2cm, font=\tiny] at (enc3_ffn-north) {FFN};""",
    to_connection("enc2_ffn", "enc3_mha"),

    # Patient Encoding - with spaced dimensions
    to_Conv("patient_enc", s_filer="", n_filer="", offset="(3,0,0)", to="(enc3_ffn-east)",
            height=25, depth=25, width=2, caption=" "),
    r"""\node[below=0.5cm, font=\small\bfseries] at (patient_enc-south) {Patient Enc};""",
    r"""\node[font=\scriptsize, rotate=90, anchor=south] at ([xshift=-0.3cm]patient_enc-west) {$N$};""",
    r"""\node[font=\scriptsize, anchor=south west] at ([yshift=0.3cm]patient_enc-near) {$128$};""",
    to_connection("enc3_ffn", "patient_enc"),

    # Patient Projection - with spaced dimensions
    to_Conv("patient_proj", s_filer="", n_filer="", offset="(3,0,0)", to="(patient_enc-east)",
            height=23, depth=23, width=2, caption=" "),
    r"""\node[below=0.5cm, font=\small\bfseries] at (patient_proj-south) {Projection};""",
    r"""\node[font=\scriptsize, rotate=90, anchor=south] at ([xshift=-0.3cm]patient_proj-west) {$N$};""",
    r"""\node[font=\scriptsize, anchor=south west] at ([yshift=0.3cm]patient_proj-near) {$128$};""",
    to_connection("patient_enc", "patient_proj"),

    # ========================================================================
    # ROW 2: AMBULANCE PROCESSING (BOTTOM) - Titles ABOVE blocks
    # ========================================================================

    # Ambulance Input - Title ABOVE, with spaced dimensions
    to_Conv("input_amb", s_filer="", n_filer="", offset="(0,-9,0)", to="(0,0,0)",
            width=2, height=18, depth=18, caption=" "),
    r"""\node[above=0.5cm, font=\small\bfseries] at (input_amb-north) {Input (Ambulances)};""",
    r"""\node[below=0.5cm, font=\scriptsize] at (input_amb-south) {$[B, V, 4]$};""",
    r"""\node[font=\scriptsize, rotate=90, anchor=south] at ([xshift=-0.3cm]input_amb-west) {$V$};""",
    r"""\node[font=\scriptsize, anchor=south west] at ([yshift=0.3cm]input_amb-near) {$4$};""",

    # Fleet Attention - Title ABOVE, with spaced dimensions
    to_Conv("fleet_att", s_filer="", n_filer="", offset="(18,-9,0)", to="(0,0,0)",
            height=18, depth=18, width=3.5, caption=" "),
    r"""\node[above=0.5cm, font=\small\bfseries] at (fleet_att-north) {Fleet Attention};""",
    r"""\node[above=1.0cm, font=\scriptsize, text=blue!70!black] at (fleet_att-north) {8 heads};""",
    r"""\node[font=\scriptsize, rotate=90, anchor=south] at ([xshift=-0.4cm]fleet_att-west) {$V$};""",
    r"""\node[font=\scriptsize, anchor=south west] at ([yshift=0.3cm]fleet_att-near) {$128$};""",
    to_connection("input_amb", "fleet_att"),

    # K,V from Patient Encoding - FIXED orientation
    r"""\draw[connection, dashed, color=blue!70, line width=0.8mm] (patient_enc-south) -- ++(0,-2) -| (fleet_att-north);""",
    r"""\node[font=\small\bfseries, text=blue!70, fill=white, inner sep=2pt] at (10, -5.5) {K, V};""",

    # Ambulance Attention - Title ABOVE, with spaced dimensions
    to_Conv("amb_att", s_filer="", n_filer="", offset="(3,0,0)", to="(fleet_att-east)",
            height=15, depth=15, width=3.5, caption=" "),
    r"""\node[above=0.5cm, font=\small\bfseries] at (amb_att-north) {Amb Attention};""",
    r"""\node[above=1.0cm, font=\scriptsize, text=blue!70!black] at (amb_att-north) {8 heads};""",
    r"""\node[font=\scriptsize, rotate=90, anchor=south] at ([xshift=-0.4cm]amb_att-west) {$1$};""",
    r"""\node[font=\scriptsize, anchor=south west] at ([yshift=0.3cm]amb_att-near) {$128$};""",
    to_connection("fleet_att", "amb_att"),

    # Ambulance Representation - Title ABOVE, with spaced dimensions
    to_Conv("amb_repr", s_filer="", n_filer="", offset="(2.5,0,0)", to="(amb_att-east)",
            height=12, depth=12, width=2, caption=" "),
    r"""\node[above=0.5cm, font=\small\bfseries] at (amb_repr-north) {Amb Repr};""",
    r"""\node[font=\scriptsize, rotate=90, anchor=south] at ([xshift=-0.3cm]amb_repr-west) {$1$};""",
    r"""\node[font=\scriptsize, anchor=south west] at ([yshift=0.3cm]amb_repr-near) {$128$};""",
    to_connection("amb_att", "amb_repr"),

    # ========================================================================
    # COMPATIBILITY SCORING (CENTER)
    # ========================================================================

    to_Sum("compat", offset="(4,0,0)", to="(patient_proj-east)", radius=2.5, opacity=0.8),
    
    to_connection("patient_proj", "compat"),
    
    # Query connection with bold label
    r"""\draw[connection, line width=0.8mm] (amb_repr-north) -- ++(0,2.5) -| (compat-south);""",
    r"""\node[font=\normalsize\bfseries, text=purple, fill=white, inner sep=2pt] at (28, -4) {Query (Q)};""",

    # ========================================================================
    # ACTOR BRANCH (TOP RIGHT) - With pointer label and spaced dimensions
    # ========================================================================
    
    # Actor label as pointer arrow
    r"""\node[font=\small\bfseries, text=green!60!black] (actor_label) at (38, 6) {Actor (Policy Network)};""",
    r"""\draw[->, thick, green!60!black] (actor_label.south) -- ++(0,-0.8);""",

    to_Conv("tanh", s_filer="", n_filer="", offset="(4,3,0)", to="(compat-east)",
            height=6, depth=18, width=1.5, caption=" "),
    r"""\node[above=0.3cm, font=\small\bfseries] at (tanh-north) {Tanh$\times C$};""",
    r"""\node[font=\scriptsize, rotate=90, anchor=south] at ([xshift=-0.2cm]tanh-west) {$N{+}1$};""",
    r"""\node[font=\scriptsize, anchor=south west] at ([yshift=0.2cm]tanh-near) {$1$};""",
    r"""\draw[connection] (compat-east) -- ++(1,0) |- (tanh-west);""",

    to_Conv("mask", s_filer="", n_filer="", offset="(2,0,0)", to="(tanh-east)",
            height=6, depth=18, width=1, caption=" "),
    r"""\node[above=0.3cm, font=\small\bfseries] at (mask-north) {Mask};""",
    r"""\node[font=\scriptsize, rotate=90, anchor=south] at ([xshift=-0.2cm]mask-west) {$N{+}1$};""",
    to_connection("tanh", "mask"),

    to_SoftMax("softmax", s_filer="", offset="(2,0,0)", to="(mask-east)", caption=" "),
    r"""\node[above=0.3cm, font=\small\bfseries] at (softmax-north) {Softmax};""",
    r"""\node[font=\scriptsize, anchor=east] at ([xshift=-0.3cm]softmax-west) {$N{+}1$};""",
    to_connection("mask", "softmax"),

    to_SoftMax("action", s_filer="", offset="(2,0,0)", to="(softmax-east)", caption=" "),
    r"""\node[above=0.3cm, font=\normalsize\bfseries, text=green!60!black] at (action-north) {$\pi(a|s)$};""",
    r"""\node[below=0.3cm, font=\small\bfseries, text=green!60!black] at (action-south) {Action};""",
    r"""\node[font=\scriptsize, anchor=east] at ([xshift=-0.3cm]action-west) {$1$};""",
    to_connection("softmax", "action"),

    # ========================================================================
    # CRITIC BRANCH (BOTTOM RIGHT) - With pointer label and spaced dimensions
    # ========================================================================
    
    # Critic label as pointer arrow
    r"""\node[font=\small\bfseries, text=red!70!black] (critic_label) at (38, -6.5) {Critic (Value Network)};""",
    r"""\draw[->, thick, red!70!black] (critic_label.north) -- ++(0,0.8);""",

    to_Conv("fc1", s_filer="", n_filer="", offset="(4,-3.5,0)", to="(compat-east)",
            height=10, depth=10, width=2.5, caption=" "),
    r"""\node[below=0.6cm, font=\small\bfseries] at (fc1-south) {FC + ReLU};""",
    r"""\node[below=1.1cm, font=\scriptsize] at (fc1-south) {$(N{+}1) \rightarrow 128$};""",
    r"""\node[font=\scriptsize, rotate=90, anchor=south] at ([xshift=-0.3cm]fc1-west) {$1$};""",
    r"""\node[font=\scriptsize, anchor=south west] at ([yshift=0.2cm]fc1-near) {$128$};""",
    r"""\draw[connection] (compat-east) -- ++(1,0) |- (fc1-west);""",

    to_Conv("fc2", s_filer="", n_filer="", offset="(2.5,0,0)", to="(fc1-east)",
            height=8, depth=8, width=2.5, caption=" "),
    r"""\node[below=0.6cm, font=\small\bfseries] at (fc2-south) {FC + ReLU};""",
    r"""\node[below=1.1cm, font=\scriptsize] at (fc2-south) {$128 \rightarrow 128$};""",
    r"""\node[font=\scriptsize, rotate=90, anchor=south] at ([xshift=-0.3cm]fc2-west) {$1$};""",
    r"""\node[font=\scriptsize, anchor=south west] at ([yshift=0.2cm]fc2-near) {$128$};""",
    to_connection("fc1", "fc2"),

    to_Conv("value", s_filer="", n_filer="", offset="(2.5,0,0)", to="(fc2-east)",
            height=5, depth=5, width=1.5, caption=" "),
    r"""\node[below=0.5cm, font=\normalsize\bfseries, text=red!70!black] at (value-south) {$V(s)$};""",
    r"""\node[above=0.3cm, font=\small\bfseries, text=red!70!black] at (value-north) {Value};""",
    r"""\node[font=\scriptsize, rotate=90, anchor=south] at ([xshift=-0.2cm]value-west) {$1$};""",
    r"""\node[font=\scriptsize, anchor=south west] at ([yshift=0.2cm]value-near) {$1$};""",
    to_connection("fc2", "value"),

    # ========================================================================
    # COMPATIBILITY FORMULA
    # ========================================================================
    r"""\node[right=0.8cm, font=\normalsize] at (compat-east) {$\displaystyle\frac{Q \cdot K^T}{\sqrt{d}}$};""",

    # ========================================================================
    # LEGEND WITH SYMBOL EXPLANATIONS
    # ========================================================================
    
    r"""
    \node[draw, rounded corners, fill=white!95, anchor=north west, align=left, font=\small, 
          minimum width=7cm, inner sep=8pt] at (-2, -13) {
        \textbf{Model Parameters}\\[0.4em]
        \begin{tabular}{ll}
        $d_{model}$ & $= 128$ (embedding dimension)\\
        Attention heads & $= 8$\\
        Encoder layers & $= 3$\\
        $d_{ff}$ & $= 512$ (feed-forward dimension)\\
        Tanh clip $C$ & $= 10.9$\\
        \end{tabular}
        \\[0.6em]
        \textbf{Notation}\\[0.4em]
        \begin{tabular}{ll}
        $B$ & Batch size\\
        $N$ & Number of patients (nodes)\\
        $V$ & Number of ambulances (vehicles)\\
        $Q$ & Query vector (current ambulance)\\
        $K, V$ & Key, Value (patient encodings)\\
        \end{tabular}
        \\[0.6em]
        \textbf{Output}\\[0.4em]
        \begin{tabular}{ll}
        \textcolor{green!60!black}{$\pi(a|s)$} & Action probabilities (policy)\\
        \textcolor{red!70!black}{$V(s)$} & State value estimate\\
        \end{tabular}
    };
    """,

    # ========================================================================
    # INPUT FEATURE ANNOTATIONS
    # ========================================================================
    r"""\node[font=\scriptsize, text=gray, anchor=west] at (-1, -5) {Features: position $(x,y)$, demand, survival time};""",
    r"""\node[font=\scriptsize, text=gray, anchor=west] at (-1, -12) {Features: position $(x,y)$, capacity, current time};""",

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()