import torch

class PVRP_Environment:
    VEH_STATE_SIZE = 5  # Added slot for earliest spoilage time
    CUST_FEAT_SIZE = 4

    def __init__(self, data, nodes=None, cust_mask=None,
                 spoilage_penalty=10.0, early_reward=0.5, unserved_penalty=10.0,
                 dist_penalty_coef=1.0, pickup_bonus_coef=5.0, idle_penalty_coef=20.0,
                 additional_late_penalty=10.0, capacity_usage_coef=0.4):
        
        # Load precomputed matrices from dataset
        self.dist_matrix = data.dist_matrix  # [batch, n+1, n+1]
        self.travel_time_matrix = data.travel_time_matrix  # [batch, n+1, n+1]
        
        # Existing initializations
        self.veh_count = data.veh_count
        self.veh_capa = data.veh_capa
        self.veh_speed = data.veh_speed
        self.nodes = data.nodes if nodes is None else nodes
        self.minibatch_size, self.nodes_count, _ = self.nodes.size()
        
        # New state tracking
        self.collected_spoilage_times = torch.full(  # Track onboard goods
            (self.minibatch_size, self.veh_count, self.veh_capa),
            float('inf'), device=self.nodes.device
        )

    def _update_vehicles(self, dest, cust_idx):
        """Enhanced vehicle update with spoilage tracking"""
        batch_idx = torch.arange(self.minibatch_size, device=self.nodes.device)
        
        # Get travel time from precomputed matrix
        travel_time = self.travel_time_matrix[batch_idx, self.cur_veh[:, 0, 0].long(), cust_idx[:, 0]]
        
        # Update collected goods spoilage times
        if cust_idx != 0:  # When picking up goods
            pickup_time = self.cur_veh[:, :, 3] + travel_time
            spoilage_time = self.nodes[batch_idx, cust_idx[:, 0], 3]
            
            # Find first empty slot in collected goods
            empty_slots = (self.collected_spoilage_times == float('inf')).float()
            slot_idx = empty_slots.argmax(dim=2)
            
            # Update collected goods
            self.collected_spoilage_times[batch_idx, self.cur_veh_idx[:, 0], slot_idx] = \
                spoilage_time - pickup_time

        # Update vehicle state
        self.cur_veh[:, :, 3] += travel_time  # Update current time
        self.cur_veh[:, :, 4] = self.collected_spoilage_times.min(dim=2)[0]  # Track earliest spoilage

    def _update_mask(self):
        """Fast feasibility check using precomputed times"""
        batch_idx = torch.arange(self.minibatch_size, device=self.nodes.device)
        current_time = self.cur_veh[:, :, 3]
        
        # Calculate latest possible departure times
        time_to_depot = self.travel_time_matrix[:, :, 0]  # [batch, nodes]
        latest_pickup_times = self.nodes[:, :, 3].unsqueeze(1) - time_to_depot.unsqueeze(1)
        
        # Time feasibility mask
        time_feasible = current_time.unsqueeze(2) <= latest_pickup_times
        
        # Capacity mask
        capacity_feasible = (self.cur_veh[:, :, 2] > 0).unsqueeze(2)
        
        # Combine masks
        self.mask = ~(time_feasible & capacity_feasible) | self.served.unsqueeze(1)
        self.mask[:, :, 0] = False  # Depot always accessible

    def step(self, cust_idx):
        """Enhanced reward calculation"""
        dest = self.nodes.gather(1, cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE))
        
        # Existing updates
        dist, lateness, ontime_pickup = self._update_vehicles(dest, cust_idx)
        
        # Dynamic spoilage penalty
        remaining_times = self.nodes[:, :, 3] - self.cur_veh[:, :, 3]
        dynamic_penalty = self.spoilage_penalty * torch.sigmoid(-remaining_times/60)
        
        # Capacity utilization reward
        used_capacity = (self.veh_capa - self.cur_veh[:, :, 2]) / self.veh_capa
        capacity_reward = self.capacity_usage_coef * used_capacity
        
        reward = (
            - self.dist_penalty_coef * dist
            + self.pickup_bonus_coef * ontime_pickup
            - dynamic_penalty * lateness
            + capacity_reward
        )
        
        # Penalize idle vehicles
        idle_penalty = -self.idle_penalty_coef * (self.cur_veh[:, :, 2] == self.veh_capa).float()
        reward += idle_penalty

        # Final updates
        self._update_done(cust_idx)
        self._update_mask()
        self._update_cur_veh()
        
        return reward

    def reset(self):
        """Initialize with precomputed matrices"""
        # Existing reset logic
        self.vehicles = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.VEH_STATE_SIZE))
        self.vehicles[:, :, :2] = self.nodes[:, 0:1, :2]  # Depot position
        self.vehicles[:, :, 2] = self.veh_capa  # Capacity
        self.vehicles[:, :, 3] = 0  # Start time
        self.vehicles[:, :, 4] = float('inf')  # No collected goods
        
        # New initialization
        self.collected_spoilage_times.fill_(float('inf'))
        
        # Existing mask initialization
        self.mask = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.nodes_count), dtype=torch.bool)
        if self.cust_mask is not None:
            self.mask = self.cust_mask[:, None, :].repeat(1, self.veh_count, 1)