# problems/_data_pvrp.py

import torch
from torch.utils.data import Dataset
import numpy as np

class PVRP_Dataset(Dataset):
    """Perishable Vehicle Routing Problem Dataset"""
    CUST_FEAT_SIZE = 4  # x, y, demand(1), spoilage_time
    QUANTILE_BINS = 10  # For quantile-based normalization

    @classmethod
    def generate(cls,
            batch_size = 1,
            cust_count = 10,
            veh_count = 2,
            veh_capa = 5,
            veh_speed = 1,
            min_cust_count = None,
            cust_loc_range = (0,101),
            horizon = 480,
            spoilage_range = (240,360),
            cluster_prob = 0.7  # Probability of spatial-temporal clustering
            ):
        size = (batch_size, cust_count, 1)

        # Generate locations with clustering
        locs, spoilage_times = cls._generate_clustered_data(
            batch_size, cust_count, cust_loc_range, 
            spoilage_range, cluster_prob
        )

        # Unit demands for all pickup points
        dems = torch.ones(size, dtype=torch.float)

        # Combine customer features
        customers = torch.cat((locs[:,1:], dems, spoilage_times), 2)

        # Add depot node
        depot_node = torch.zeros((batch_size, 1, cls.CUST_FEAT_SIZE))
        depot_node[:,:,:2] = locs[:,0:1]
        nodes = torch.cat((depot_node, customers), 1)

        # Create distance and time matrices
        dist_matrix = torch.cdist(locs, locs)  # [batch, n+1, n+1]
        travel_time_matrix = dist_matrix / veh_speed

        if min_cust_count is not None:
            counts = torch.randint(min_cust_count+1, cust_count+2, (batch_size, 1), dtype=torch.int64)
            cust_mask = torch.arange(cust_count+1).expand(batch_size, -1) > counts
            nodes[cust_mask] = 0
        else:
            cust_mask = None

        dataset = cls(veh_count, veh_capa, veh_speed, nodes, cust_mask)
        dataset.dist_matrix = dist_matrix
        dataset.travel_time_matrix = travel_time_matrix
        return dataset

    
    @staticmethod
    def _generate_clustered_data(batch_size, cust_count, loc_range, spoilage_range, cluster_prob):
        """Generate clustered data with proper bounds checking"""
        locs = torch.zeros((batch_size, cust_count+1, 2))
        spoilage_times = torch.zeros((batch_size, cust_count, 1))
        
        for b in range(batch_size):
            # Generate depot location
            depot = torch.FloatTensor(2).uniform_(*loc_range)
            locs[b,0] = depot
            
            # Create 2-4 clusters per instance
            num_clusters = torch.randint(2, 5, (1,)).item()
            cluster_centers = []
            time_clusters = []
            
            # Generate cluster centers and times
            for _ in range(num_clusters):
                center = torch.FloatTensor(2).uniform_(*loc_range)
                time = torch.FloatTensor(1).uniform_(*spoilage_range)
                cluster_centers.append(center)
                time_clusters.append(time)
            
            # Assign customers to clusters
            for c in range(1, cust_count+1):
                if torch.rand(1) < cluster_prob and num_clusters > 0:
                    # Assign to random cluster
                    cluster_idx = torch.randint(0, num_clusters, (1,)).item()
                    center = cluster_centers[cluster_idx]
                    time = time_clusters[cluster_idx] + torch.randn(1).clamp(-2,2)*20
                else:
                    # Assign randomly
                    center = depot
                    time = torch.FloatTensor(1).uniform_(*spoilage_range)
                
                # Generate location with bounds checking
                loc = center + torch.randn(2) * 15
                loc = torch.clamp(loc, *loc_range)
                
                # Assign values
                locs[b,c] = loc
                spoilage_times[b,c-1] = torch.clamp(time, *spoilage_range)
                    
        return locs, spoilage_times

    def __init__(self, veh_count, veh_capa, veh_speed, nodes, cust_mask=None):
        """Initialize PVRP dataset"""
        self.veh_count = veh_count
        self.veh_capa = veh_capa
        self.veh_speed = veh_speed
        self.nodes = nodes
        self.batch_size, self.nodes_count, d = nodes.size()
        self.cust_mask = cust_mask
        self.dist_matrix = None
        self.travel_time_matrix = None

        if d != self.CUST_FEAT_SIZE:
            raise ValueError(f"Expected {self.CUST_FEAT_SIZE} features, got {d}")

    def normalize(self):
        """Quantile-based normalization"""
        # Location normalization
        loc_min = self.nodes[:,:,:2].amin(dim=(0,1))
        loc_max = self.nodes[:,:,:2].amax(dim=(0,1))
        loc_range = loc_max - loc_min
        self.nodes[:,:,:2] = (self.nodes[:,:,:2] - loc_min) / loc_range

        # Spoilage time quantization
        sorted_times = torch.sort(self.nodes[:,1:,3].flatten())[0]
        quantiles = torch.linspace(0, 1, self.QUANTILE_BINS+1)
        self.time_quantiles = torch.quantile(sorted_times, quantiles)
        
        # Apply quantile normalization
        for b in range(self.batch_size):
            for n in range(1, self.nodes_count):
                if self.nodes[b,n,3] > 0:  # Skip masked customers
                    q_idx = torch.searchsorted(self.time_quantiles, self.nodes[b,n,3]) - 1
                    self.nodes[b,n,3] = q_idx / self.QUANTILE_BINS

        # Normalize distance matrices
        if self.dist_matrix is not None:
            self.dist_matrix = self.dist_matrix / loc_range.max()
            self.travel_time_matrix = self.dist_matrix / self.veh_speed

        return loc_range.max(), self.time_quantiles[-1]

    def save(self, fpath):
        """Save dataset with matrices"""
        torch.save({
            "veh_count": self.veh_count,
            "veh_capa": self.veh_capa,
            "veh_speed": self.veh_speed,
            "nodes": self.nodes,
            "cust_mask": self.cust_mask,
            "dist_matrix": self.dist_matrix,
            "travel_time_matrix": self.travel_time_matrix,
            "time_quantiles": getattr(self, 'time_quantiles', None)
        }, fpath)

    @classmethod
    def load(cls, fpath):
        """Load dataset with matrices"""
        data = torch.load(fpath)
        dataset = cls(data["veh_count"], data["veh_capa"], 
                     data["veh_speed"], data["nodes"], data["cust_mask"])
        dataset.dist_matrix = data["dist_matrix"]
        dataset.travel_time_matrix = data["travel_time_matrix"]
        if "time_quantiles" in data:
            dataset.time_quantiles = data["time_quantiles"]
        return dataset

    # Existing methods remain unchanged
    def __len__(self): return self.batch_size
    def __getitem__(self, i): return (self.nodes[i], self.cust_mask[i]) if self.cust_mask else self.nodes[i]
    def nodes_gen(self): yield from (n[m^1] if m is not None else n for n,m in zip(self.nodes, self.cust_mask))