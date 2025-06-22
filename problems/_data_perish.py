# problems/_data_pvrp.py

import torch
from torch.utils.data import Dataset

class PVRP_Dataset(Dataset):
    """Perishable Vehicle Routing Problem Dataset"""
    CUST_FEAT_SIZE = 4  # x, y, demand(1), spoilage_time

    @classmethod
    def generate(cls,
            batch_size = 1,
            cust_count = 10,
            veh_count = 2,
            veh_capa = 5,
            veh_speed = 2,
            min_cust_count = None,
            cust_loc_range = (0,101),
            horizon = 480,
            spoilage_range = (360,460)  # Time window for spoilage
            ):
        size = (batch_size, cust_count, 1)

        # Sample locations (x_j, y_j ~ U(0, 100))
        locs = torch.randint(*cust_loc_range, (batch_size, cust_count+1, 2), dtype=torch.float)
        
        # Unit demands for all pickup points
        dems = torch.ones(size, dtype=torch.float)

        # Sample spoilage times - independent of location/travel time
        # spoilage_times = torch.randint(*spoilage_range, size, dtype=torch.float)
        spoilage_times = torch.randint(*spoilage_range, size, dtype=torch.int64)

        # Combine customer features
        customers = torch.cat((locs[:,1:], dems, spoilage_times), 2)

        # Add depot node
        depot_node = torch.zeros((batch_size, 1, cls.CUST_FEAT_SIZE))
        depot_node[:,:,:2] = locs[:,0:1]
        nodes = torch.cat((depot_node, customers), 1)

        if min_cust_count is not None:
            counts = torch.randint(min_cust_count+1, cust_count+2, (batch_size, 1), dtype=torch.int64)
            cust_mask = torch.arange(cust_count+1).expand(batch_size, -1) > counts
            nodes[cust_mask] = 0
        else:
            cust_mask = None

        dataset = cls(veh_count, veh_capa, veh_speed, nodes, cust_mask)
        return dataset

    def __init__(self, veh_count, veh_capa, veh_speed, nodes, cust_mask=None):
        """Initialize PVRP dataset
        
        Args:
            veh_count: Number of vehicles
            veh_capa: Vehicle capacity
            veh_speed: Vehicle speed
            nodes: Node features tensor
            cust_mask: Optional mask for variable size instances
        """
        self.veh_count = veh_count
        self.veh_capa = veh_capa
        self.veh_speed = veh_speed

        self.nodes = nodes
        self.batch_size, self.nodes_count, d = nodes.size()
        if d != self.CUST_FEAT_SIZE:
            raise ValueError(f"Expected {self.CUST_FEAT_SIZE} customer features per node, got {d}")
        self.cust_mask = cust_mask

    def __len__(self):
        return self.batch_size

    def __getitem__(self, i):
        if self.cust_mask is None:
            return self.nodes[i]
        else:
            return self.nodes[i], self.cust_mask[i]

    def nodes_gen(self):
        if self.cust_mask is None:
            yield from self.nodes
        else:
            yield from (n[m^1] for n,m in zip(self.nodes, self.cust_mask))

    def normalize(self):
        # Normalize locations
        loc_scl, loc_off = self.nodes[:,:,:2].max().item(), self.nodes[:,:,:2].min().item()
        loc_scl -= loc_off
        
        self.nodes[:,:,:2] -= loc_off
        self.nodes[:,:,:2] /= loc_scl
        
        # Normalize spoilage times
        time_scl = self.nodes[:,:,3].max().item()
        self.nodes[:,:,3] /= time_scl
        
        # No need to normalize demands as they are unit demands
        #self.veh_capa = 1
        #self.veh_speed *= time_scl / loc_scl
        
        return loc_scl, time_scl

    def save(self, fpath):
        """Save dataset to file"""
        torch.save({
            "veh_count": self.veh_count,
            "veh_capa": self.veh_capa,
            "veh_speed": self.veh_speed,
            "nodes": self.nodes,
            "cust_mask": self.cust_mask
        }, fpath)

    @classmethod
    def load(cls, fpath):
        """Load dataset from file"""
        return cls(**torch.load(fpath))