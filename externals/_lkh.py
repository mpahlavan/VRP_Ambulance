from marpdan.dep import LKH_ENABLED, LKH_BIN
from marpdan.dep import tqdm
from multiprocessing import Pool
import subprocess
import tempfile
import os.path
import time


def _call_lkh(nodes, veh_count, veh_capa, prefix="/tmp/mardan_lkh0"):
    try:
        # Create TSP file
        tsp_path = f"{prefix}.tsp"
        par_path = f"{prefix}.par"
        tr_path = f"{prefix}.tour"
        
        with open(tsp_path, 'w') as tsp_f:
            tsp_f.write("NAME : temp\n")  # Note the space after :
            tsp_f.write("TYPE : CVRP\n")
            tsp_f.write("DIMENSION : {}\n".format(nodes.size(0)))
            tsp_f.write("VEHICLES : {}\n".format(veh_count))
            tsp_f.write("CAPACITY : {}\n".format(veh_capa))
            tsp_f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
            tsp_f.write("NODE_COORD_TYPE : TWOD_COORDS\n")
            
            # Node coordinates
            tsp_f.write("NODE_COORD_SECTION\n")
            for j, (x, y) in enumerate(nodes[:,:2], start=1):
                tsp_f.write(f"{j} {x:.0f} {y:.0f}\n")
            
            # Demands (unit demand for PVRP)
            tsp_f.write("DEMAND_SECTION\n")
            for j in range(1, nodes.size(0) + 1):
                demand = 0 if j == 1 else 1  # 0 for depot, 1 for customers
                tsp_f.write(f"{j} {demand}\n")
            
            # Depot
            tsp_f.write("DEPOT_SECTION\n")
            tsp_f.write("1\n")
            tsp_f.write("-1\n")
            
            tsp_f.write("EOF\n")

        # Write parameter file
        with open(par_path, "w") as par_f:
            par_f.write("PROBLEM_FILE = {}\n".format(tsp_path))
            par_f.write("TOUR_FILE = {}\n".format(tr_path))
            par_f.write("MTSP_SOLUTION_FILE = {}\n".format(tr_path))
            par_f.write("MAX_TRIALS = 1000\n")
            par_f.write("RUNS = 1\n")
            par_f.write("TRACE_LEVEL = 0\n")
            par_f.write("SEED = 1\n")

        # Run LKH
        result = subprocess.run(
            [LKH_BIN, par_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False  # Don't raise exception on non-zero exit
        )

        if result.returncode != 0:
            print(f"LKH stderr: {result.stderr.decode()}")
            return [[0]]  # Return fallback route

        # Check if tour file exists
        if not os.path.exists(tr_path):
            print(f"Tour file not created: {tr_path}")
            return [[0]]

        # Read solution
        with open(tr_path, 'r') as tr_f:
            lines = tr_f.readlines()

        # Parse routes
        routes = []
        current_route = []
        reading_tour = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('TOUR_SECTION'):
                reading_tour = True
                current_route = []
            elif line == '-1':
                if current_route:
                    routes.append(current_route)
                current_route = []
            elif reading_tour and line.isdigit():
                node = int(line) - 1
                if node >= 0:
                    current_route.append(node)

        if current_route:
            routes.append(current_route)

        if not routes:
            print("No valid routes found in solution")
            return [[0]]

        return routes

    except Exception as e:
        print(f"Error in _call_lkh: {str(e)}")
        return [[0]]

def lkh_solve(data):
    """Solve PVRP instances using LKH"""
    with Pool() as p:
        with tqdm(desc="Calling LKH", total=data.batch_size) as pbar:
            with tempfile.TemporaryDirectory(prefix="mardan_lkh") as tmp_dir:
                results = []
                for b, nodes in enumerate(data.nodes_gen()):
                    prefix = os.path.join(tmp_dir, str(b))
                    result = p.apply_async(
                        _call_lkh,
                        (nodes, data.veh_count, data.veh_capa, prefix),
                        callback=lambda _: pbar.update()
                    )
                    results.append(result)
                
                routes = []
                for res in results:
                    try:
                        route = res.get(timeout=30)
                        routes.append(route)
                    except Exception as e:
                        print(f"Error getting result: {str(e)}")
                        routes.append([[0]])
                        
    return routes