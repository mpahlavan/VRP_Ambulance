import torch

class PerishableVRP_Environment:
    VEH_STATE_SIZE = 5  # position(2), time(1), capacity(1), urgent_deadline(1)
    CUST_FEAT_SIZE = 4  # x,y, demand(1), spoilage_time

    def __init__(self, data, nodes=None, cust_mask=None,
                 spoilage_penalty=2.0, early_reward=0.5, 
                 unserved_penalty=1.0):
        """Initialize environment
        Args:
            data: Dataset object containing problem instance
            nodes: Optional custom nodes tensor
            cust_mask: Optional custom customer mask
            spoilage_penalty: Penalty multiplier for late deliveries
            early_reward: Reward multiplier for early deliveries
            unserved_penalty: Penalty for unserved nodes
        """
        self.veh_count = data.veh_count
        self.veh_capa = data.veh_capa
        self.veh_speed = data.veh_speed
        self.nodes = data.nodes if nodes is None else nodes
        self.init_cust_mask = data.cust_mask if cust_mask is None else cust_mask
        self.minibatch_size, self.nodes_count, _ = self.nodes.size()
        
        self.spoilage_penalty = spoilage_penalty
        self.early_reward = early_reward
        self.unserved_penalty = unserved_penalty

    def _calculate_travel_time(self, from_pos, to_pos):
        """Calculate travel time between positions"""
        dist = torch.pairwise_distance(from_pos, to_pos)
        return dist / self.veh_speed

    def _check_feasibility(self, current_pos, next_pos, depot_pos, current_time, deadline):
        """Check if pickup is feasible considering depot return
        Returns True if goods can be delivered before deadline"""
        time_to_next = self._calculate_travel_time(current_pos, next_pos)
        time_to_depot = self._calculate_travel_time(next_pos, depot_pos)
        return current_time + time_to_next + time_to_depot <= deadline

    def _update_vehicles(self, dest):
        """Update vehicle states after movement
        Updates position, capacity, time, and deadline"""
        # Calculate travel time
        dist = torch.pairwise_distance(self.cur_veh[:, 0, :2], dest[:, 0, :2], keepdim=True)
        travel_time = dist / self.veh_speed

        # Update vehicle state
        self.cur_veh[:, :, :2] = dest[:, :, :2]  # Update position
        self.cur_veh[:, :, 2] += travel_time     # Update time
        self.cur_veh[:, :, 3] -= 1.0             # Update capacity (always 1 unit)
        
        # Update urgent deadline (min of current and new pickup)
        new_deadline = dest[:, :, 3]  # Spoilage time of new pickup
        self.cur_veh[:, :, 4] = torch.min(self.cur_veh[:, :, 4], new_deadline)
        
        # Update global vehicle states
        self.vehicles = self.vehicles.scatter(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE), 
            self.cur_veh)
        
        return dist

    def _update_mask(self, cust_idx):
        """Update mask for feasible pickups"""
        self.served.scatter_(1, cust_idx, cust_idx > 0)
        
        # Mask for capacity constraints
        capacity_mask = (self.cur_veh[:, :, 3] <= 0)
        
        # Mask for feasibility (can reach depot before deadline)
        depot_pos = self.nodes[:, 0:1, :2]  # Depot position
        time_mask = torch.zeros_like(self.mask, dtype=torch.bool)
        
        for i in range(self.nodes_count):
            if i == 0:  # Skip depot
                continue
            node_pos = self.nodes[:, i:i+1, :2]
            node_deadline = self.nodes[:, i:i+1, 3]
            feasible = self._check_feasibility(
                self.cur_veh[:, :, :2],
                node_pos,
                depot_pos,
                self.cur_veh[:, :, 2],
                node_deadline
            )
            time_mask[:, :, i] = ~feasible

        # Combine all masks
        self.mask = self.mask | self.served[:, None, :] | capacity_mask[:, :, None] | time_mask
        self.mask[:, :, 0] = 0  # Depot always available
        
    def _calculate_rewards(self, travel_dist):
        """Calculate rewards/penalties for current step"""
        rewards = -travel_dist  # Basic distance cost
        
        if self.done:
            # Calculate spoilage penalties
            '''arrival_times = self.vehicles[:, :, 2]  # Arrival times at depot
            
            print(arrival_times.size(0), deadlines.size(0))
            '''
            # Get vehicle arrival times: [batch_size, n_vehicles]
            vehicle_times = self.vehicles[:, :, 2]
            # Initialize arrival times tensor for all nodes
            arrival_times = self.nodes.new_zeros((self.minibatch_size, self.nodes_count))

            # Ensure the sizes of deadlines and arrival_times match
            # For each vehicle, scatter its arrival time to the nodes it served
            for v in range(self.veh_count):
                vehicle_mask = (self.vehicle_routes == v)  # nodes served by vehicle v
                arrival_times.masked_scatter_(
                    vehicle_mask,
                    vehicle_times[:, v].unsqueeze(1).expand(-1, vehicle_mask.size(1))
                )
            
            deadlines = self.nodes[:, :, 3]        # Spoilage deadlines customer
           
            # Calculate early delivery rewards [batch_size, n_nodes]
            early_delivery = torch.clamp(deadlines - arrival_times, min=0)
            early_delivery = early_delivery * self.served.float()  # Only count served nodes
            # Sum across nodes dimension and keep batch dimension [batch_size, 1]
            early_reward = self.early_reward * early_delivery.sum(dim=1, keepdim=True)
            
            # Calculate late delivery penalties [batch_size, n_nodes]
            late_delivery = torch.clamp(arrival_times - deadlines, min=0)
            late_delivery = late_delivery * self.served.float()  # Only count served nodes
            # Sum across nodes dimension and keep batch dimension [batch_size, 1]
            late_penalty = self.spoilage_penalty * late_delivery.sum(dim=1, keepdim=True)
            
            
            # Unserved nodes penalty
            if self.init_cust_mask is not None:
                self.served += self.init_cust_mask
            unserved = (self.served ^ True).float().sum(-1, keepdim=True) - 1
            rewards -= self.unserved_penalty * unserved
            
        return rewards

    def reset(self):
        """Reset environment to initial state"""
        # Initialize vehicle states: pos(2), time(1), capacity(1), deadline(1)
        self.vehicles = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.VEH_STATE_SIZE))
        self.vehicles[:, :, :2] = self.nodes[:, 0:1, :2]  # Start at depot
        self.vehicles[:, :, 3] = self.veh_capa           # Full capacity
        self.vehicles[:, :, 4] = float('inf')            # Initial deadline

        # Initialize masks and flags
        self.veh_done = self.nodes.new_zeros((self.minibatch_size, self.veh_count), dtype=torch.bool)
        self.done = False
        self.cust_mask = self.init_cust_mask
        self.served = self.nodes.new_zeros((self.minibatch_size, self.nodes_count), dtype=torch.bool)
        
        # Initialize masks for feasible actions
        self.mask = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.nodes_count), dtype=torch.bool)
        if self.cust_mask is not None:
            self.mask = self.cust_mask[:, None, :].repeat(1, self.veh_count, 1)
            
        # Set current vehicle
        self.cur_veh_idx = self.nodes.new_zeros((self.minibatch_size, 1), dtype=torch.int64)
        self.cur_veh = self.vehicles.gather(1, 
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE))
        self.cur_veh_mask = self.mask.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))
        
        # Initialize vehicle routes tracker
        self.vehicle_routes = self.nodes.new_zeros(
            (self.minibatch_size, self.nodes_count),
            dtype=torch.long
        ) - 1  # -1 indicates node not served by any vehicle yet

    def step(self, cust_idx):
        """Execute one environment step"""
        # Get destination info
        dest = self.nodes.gather(1, 
            cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE))
        
        # Update vehicle states and get travel distance
        travel_dist = self._update_vehicles(dest)
        
        # Update completion status
        self._update_done(cust_idx)
        
        # Update action masks
        self._update_mask(cust_idx)
        
        # Update current vehicle
        self._update_cur_veh()
        
        # Calculate rewards
        rewards = self._calculate_rewards(travel_dist)
        
        # Update vehicle routes
        self.vehicle_routes.scatter_(
            1,
            cust_idx,
            self.cur_veh_idx.expand(-1, cust_idx.size(1))
        )
        return rewards

    def _update_done(self, cust_idx):
        """Update completion status"""
        self.veh_done.scatter_(1, self.cur_veh_idx, cust_idx == 0)
        self.done = bool(self.veh_done.all())

    def _update_cur_veh(self):
        """Update current vehicle selection"""
        avail = self.vehicles[:, :, 2].clone()  # Use time as priority
        avail[self.veh_done] = float('inf')
        self.cur_veh_idx = avail.argmin(1, keepdim=True)
        self.cur_veh = self.vehicles.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE))
        self.cur_veh_mask = self.mask.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))