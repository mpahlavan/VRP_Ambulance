#!/usr/bin/env python3
"""
PlotNeuralNet visualization for PVRP Actor-Critic Architecture
Based on the AttentionLearner and CriticBaseline implementation

To use this script:
1. Clone PlotNeuralNet: git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
2. Place this file in PlotNeuralNet/pyexamples/
3. Run: cd pyexamples && python pvrp_architecture.py
4. Compile: pdflatex pvrp_architecture.tex

Architecture Components:
- Customer Embedding (depot + customers)
- Transformer Encoder (3 layers x 8 heads)
- Fleet Attention (vehicles attend to customer encodings)
- Vehicle Attention (current vehicle query)
- Compatibility scoring
- Actor (softmax policy) and Critic (MLP value estimation)
"""

import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *


# ============================================================================
# Main Architecture Definition
# ============================================================================

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # ========================================================================
    # INPUT SECTION
    # ========================================================================

    # Customer Input [B, N, 4]
    to_Conv("input_cust", s_filer=50, n_filer=4, offset="(0,0,0)", to="(0,0,0)",
            width=2, height=40, depth=40, caption="Customers"),

    # Vehicle Input [B, V, 4]
    to_Conv("input_veh", s_filer=5, n_filer=4, offset="(0,-8,0)", to="(0,0,0)",
            width=2, height=25, depth=25, caption="Vehicles"),

    # ========================================================================
    # EMBEDDING SECTION
    # ========================================================================

    # Depot Embedding
    to_Conv("depot_emb", s_filer=1, n_filer=128, offset="(2.5,2,0)", to="(input_cust-east)",
            height=8, depth=8, width=2, caption="Depot Emb"),

    # Customer Embedding
    to_Conv("cust_emb", s_filer=50, n_filer=128, offset="(2.5,-0.5,0)", to="(input_cust-east)",
            height=35, depth=35, width=2, caption="Cust Emb"),

    to_connection("input_cust", "cust_emb"),

    # Concatenate embeddings
    to_Conv("concat_emb", s_filer=51, n_filer=128, offset="(2,0,0)", to="(cust_emb-east)",
            height=40, depth=40, width=2, caption="Concat"),

    to_connection("depot_emb", "concat_emb"),
    to_connection("cust_emb", "concat_emb"),

    # ========================================================================
    # TRANSFORMER ENCODER SECTION
    # ========================================================================

    # Transformer Encoder Layer 1
    to_ConvConvRelu("tf_layer1", s_filer=51, n_filer=(8, 512), offset="(2,0,0)", to="(concat_emb-east)",
            height=38, depth=38, width=(2,2), caption="TF Layer 1"),
    to_connection("concat_emb", "tf_layer1"),

    # Transformer Encoder Layer 2
    to_ConvConvRelu("tf_layer2", s_filer=51, n_filer=(8, 512), offset="(1.5,0,0)", to="(tf_layer1-east)",
            height=36, depth=36, width=(2,2), caption="TF Layer 2"),
    to_connection("tf_layer1", "tf_layer2"),

    # Transformer Encoder Layer 3
    to_ConvConvRelu("tf_layer3", s_filer=51, n_filer=(8, 512), offset="(1.5,0,0)", to="(tf_layer2-east)",
            height=34, depth=34, width=(2,2), caption="TF Layer 3"),
    to_connection("tf_layer2", "tf_layer3"),

    # Customer Encoding output
    to_Conv("cust_enc", s_filer=51, n_filer=128, offset="(2,0,0)", to="(tf_layer3-east)",
            height=32, depth=32, width=2, caption="Cust Enc"),
    to_connection("tf_layer3", "cust_enc"),

    # ========================================================================
    # FLEET ATTENTION SECTION
    # ========================================================================

    # Fleet Attention - vehicles attend to customer encodings
    to_ConvConvRelu("fleet_att", s_filer=5, n_filer=(8, 128), offset="(3,-4,0)", to="(cust_enc-east)",
            height=25, depth=25, width=(2,2), caption="Fleet Att"),

    # Connection from vehicles to fleet attention
    to_connection("input_veh", "fleet_att"),

    # Dashed connection from customer encoding to fleet attention (K, V)
    r"""\draw[connection, dashed] (cust_enc-east) -- ++(1.5,0) |- (fleet_att-north);""",

    # ========================================================================
    # VEHICLE ATTENTION SECTION
    # ========================================================================

    # Vehicle Attention - current vehicle queries fleet representation
    to_ConvConvRelu("veh_att", s_filer=1, n_filer=(8, 128), offset="(2,0,0)", to="(fleet_att-east)",
            height=20, depth=20, width=(2,2), caption="Veh Att"),
    to_connection("fleet_att", "veh_att"),

    # Vehicle Representation output
    to_Conv("veh_repr", s_filer=1, n_filer=128, offset="(2,0,0)", to="(veh_att-east)",
            height=15, depth=15, width=2, caption="Veh Repr"),
    to_connection("veh_att", "veh_repr"),

    # ========================================================================
    # CUSTOMER PROJECTION
    # ========================================================================

    # Customer Projection
    to_Conv("cust_proj", s_filer=51, n_filer=128, offset="(5,4,0)", to="(cust_enc-east)",
            height=30, depth=30, width=2, caption="Cust Proj"),
    to_connection("cust_enc", "cust_proj"),

    # ========================================================================
    # COMPATIBILITY SCORING
    # ========================================================================

    # Matmul for compatibility
    to_Sum("compat", offset="(5,0,0)", to="(cust_proj-east)", radius=2.5),

    to_connection("cust_proj", "compat"),

    # Connection from vehicle representation
    r"""\draw[connection] (veh_repr-east) -- ++(1,0) |- (compat-south);""",

    # ========================================================================
    # ACTOR HEAD (POLICY)
    # ========================================================================

    # Tanh scaling
    to_Conv("tanh_scale", s_filer=51, n_filer=1, offset="(2,1.5,0)", to="(compat-east)",
            height=12, depth=25, width=1, caption="Tanh"),

    r"""\draw[connection] (compat-east) -- ++(0.5,0) |- (tanh_scale-west);""",

    # Softmax for action probabilities
    to_SoftMax("softmax", s_filer=51, offset="(1.5,0,0)", to="(tanh_scale-east)",
               caption="Softmax"),
    to_connection("tanh_scale", "softmax"),

    # Policy output
    to_SoftMax("policy_out", s_filer=51, offset="(1.5,0,0)", to="(softmax-east)",
               caption="Action"),
    to_connection("softmax", "policy_out"),

    # ========================================================================
    # CRITIC HEAD (VALUE)
    # ========================================================================

    # Critic MLP
    to_Conv("critic_mlp1", s_filer=1, n_filer=128, offset="(2,-1.5,0)", to="(compat-east)",
            height=10, depth=10, width=2, caption="FC+ReLU"),

    r"""\draw[connection] (compat-east) -- ++(0.5,0) |- (critic_mlp1-west);""",

    to_Conv("critic_mlp2", s_filer=1, n_filer=128, offset="(1.5,0,0)", to="(critic_mlp1-east)",
            height=8, depth=8, width=2, caption="FC+ReLU"),
    to_connection("critic_mlp1", "critic_mlp2"),

    to_Conv("critic_out", s_filer=1, n_filer=1, offset="(1.5,0,0)", to="(critic_mlp2-east)",
            height=5, depth=5, width=1, caption="Value"),
    to_connection("critic_mlp2", "critic_out"),

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
