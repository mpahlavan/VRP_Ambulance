import torch
class PVRP_Environment:
    VEH_STATE_SIZE = 4  # position(2), capacity(1), time(1)
    CUST_FEAT_SIZE = 4  # x,y, demand(1), spoilage_time

    # 4 feature:
    #x-coordinate of the vehicle's location:cur_veh[:, :, 0]
    #y-coordinate of the vehicle's location: cur_veh[:, :, 1]
    #capacity of the vehicles:cur_veh[:, :, 2]
    #available time of the vehicles:cur_veh[:, :, 3]


    def __init__(self, data, nodes=None, cust_mask=None,
             spoilage_penalty=10.0, early_reward=0.5, unserved_penalty=10.0,
             dist_penalty_coef=1.0, pickup_bonus_coef=5.0, idle_penalty_coef=20.0,
             additional_late_penalty=10.0, capacity_usage_coef=0.4):
       self.veh_count = data.veh_count
       self.veh_capa = data.veh_capa
       self.veh_speed = data.veh_speed
       self.nodes = data.nodes if nodes is None else nodes
       self.init_cust_mask = data.cust_mask if cust_mask is None else cust_mask
       self.minibatch_size, self.nodes_count, _ = self.nodes.size()
       
       self.spoilage_penalty = spoilage_penalty
       self.early_reward = early_reward
       self.unserved_penalty = unserved_penalty
       self.dist_penalty_coef = dist_penalty_coef
       self.pickup_bonus_coef = pickup_bonus_coef
       self.idle_penalty_coef = idle_penalty_coef
       self.additional_late_penalty = additional_late_penalty
       self.capacity_usage_coef = capacity_usage_coef

       
    def _update_vehicles(self, dest, cust_idx):
        """Update vehicle states after moving to destination"""
        # Calculate travel distance and time
        dist = torch.pairwise_distance(
            self.cur_veh[:, 0, :2], 
            dest[:, 0, :2], 
            keepdim=True
        )
        travel_time = dist / self.veh_speed
        arrival_time = self.cur_veh[:, :, 3] + travel_time

        # Calculate time to return to depot
        to_depot_dist = torch.pairwise_distance(
            dest[:, 0, :2], 
            self.nodes[:, 0, :2], 
            keepdim=True
        )
        to_depot_time = to_depot_dist / self.veh_speed
        latest_arrival = dest[:, :, 3] - to_depot_time

        # Check if delivery was late (convert to boolean first)
        lateness = (arrival_time > latest_arrival).bool()
        not_late = ~lateness
        is_customer = (cust_idx != 0).bool()
        ontime_pickup = (not_late & is_customer).float()
        lateness = lateness.float()

        # Track late nodes
        if cust_idx.any():
            self.late_nodes.scatter_(1, cust_idx, lateness > 0)
            self.node_lateness_count += lateness.sum()

        # Update vehicle state
        self.cur_veh[:, :, :2] = dest[:, :, :2]  # Position x,y
        self.cur_veh[:, :, 2] -= 1.0             # Capacity
        self.cur_veh[:, :, 3] += travel_time     # Time

        # Update vehicles tensor
        self.vehicles = self.vehicles.scatter(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE),
            self.cur_veh)

        return dist, lateness, ontime_pickup
    
   
    def _update_done(self, cust_idx):
       self.veh_done.scatter_(1, self.cur_veh_idx, cust_idx == 0)
       self.done = bool(self.veh_done.all())

    '''
    def _update_mask(self, cust_idx):
       # Mark served customers
       self.served.scatter_(1, cust_idx, cust_idx > 0)

       # Update urgent deadlines for current vehicle
       if cust_idx.any():
           new_deadlines = self.nodes.gather(1, cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE))[:, :, 3]
           self.urgent_deadlines.scatter_(1, self.cur_veh_idx, 
               torch.min(self.urgent_deadlines.gather(1, self.cur_veh_idx), new_deadlines))

       # Calculate travel times
       depot_pos = self.nodes[:, 0:1, :2]  # [batch, 1, 2]
       node_pos = self.nodes[:, :, :2]     # [batch, nodes, 2]
       current_time = self.cur_veh[:, :, 3] # [batch, 1]

       # Calculate distances and times with proper dimensions
       to_node_dist = (node_pos.unsqueeze(1) - self.cur_veh[:, :, None, :2]).norm(dim=-1)  # [batch, veh, nodes]
       to_node_time = to_node_dist / self.veh_speed
       to_depot_dist = (node_pos - depot_pos).norm(dim=-1)  # [batch, nodes]
       to_depot_time = to_depot_dist.unsqueeze(1) / self.veh_speed  # [batch, 1, nodes]
       
       # Calculate total time [batch, veh, nodes]
       total_time = current_time.unsqueeze(-1) + to_node_time + to_depot_time

       # Time mask for current vehicle
       spoilage_times = self.nodes[:, :, 3].unsqueeze(1)  # [batch, 1, nodes]
       time_constraints = (total_time > spoilage_times)    # [batch, veh, nodes]
       
       # Create time mask only for current vehicle
       time_mask = torch.zeros_like(self.mask)  # [batch, veh, nodes]
       time_mask.scatter_(1, 
           self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count),
           time_constraints.gather(1, self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count)))

       # Capacity constraint
       capacity_mask = torch.zeros_like(self.mask).scatter_(1,
           self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count),
           self.cur_veh[:, :, None, 2] < 1)

       # Combine masks
       self.mask = (self.served.unsqueeze(1) |
                   capacity_mask |
                   time_mask |
                   self.veh_done.unsqueeze(-1))
       self.mask[:, :, 0] = 0  # Depot always available

       # Update current vehicle mask
       self.cur_veh_mask = self.mask.gather(1,
           self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))
    '''
   
    def _update_mask(self, cust_idx):
        self.new_customers = False
        # Mark served customers
        self.served.scatter_(1, cust_idx, cust_idx > 0)
        
        # Vehicle state [batch, 1, 2/1]
        active_veh_pos = self.cur_veh[:, :, :2]
        active_veh_time = self.cur_veh[:, :, 3].unsqueeze(-1)  # Add dimension for broadcasting
        
        # Positions [batch, nodes, 2] and [batch, 1, 2]
        depot_pos = self.nodes[:, 0:1, :2]
        node_pos = self.nodes[:, :, :2]
        
        # Time calculations [batch, nodes]
        to_node_dist = torch.cdist(active_veh_pos, node_pos)
        to_node_time = to_node_dist.squeeze(1) / self.veh_speed
        
        to_depot_dist = torch.cdist(node_pos, depot_pos)
        to_depot_time = to_depot_dist.squeeze(-1) / self.veh_speed
        
        # Track carried goods [batch, nodes]
        carrying_mask = torch.zeros_like(self.served, dtype=torch.bool)
        for b in range(self.minibatch_size):
            active_veh_id = self.cur_veh_idx[b].item()
            carrying_mask[b] = (self.vehicle_routes[b] == active_veh_id) & self.served[b]
        
        # Get minimum spoilage times [batch, 1]
        min_spoilage = torch.full((self.minibatch_size, 1), float('inf'), 
                                device=self.nodes.device)
        for b in range(self.minibatch_size):
            carried_goods = self.nodes[b, carrying_mask[b]]
            if len(carried_goods) > 0:
                min_spoilage[b] = carried_goods[:, 3].min()
        
        # Total time calculations [batch, nodes]
        total_time = active_veh_time.squeeze(1) + to_node_time + to_depot_time
        
        # Create spoilage mask [batch, nodes]
        active_veh_spoilage = (total_time > min_spoilage)
        carrying_any = carrying_mask.any(dim=1, keepdim=True)
        active_veh_spoilage = active_veh_spoilage & carrying_any
        
        # Initialize full mask [batch, veh_count, nodes]
        spoilage_mask = torch.zeros_like(self.mask)
        spoilage_mask.scatter_(1, 
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count),
            active_veh_spoilage.unsqueeze(1))
        
        # Capacity constraints [batch, veh_count, nodes]
        capacity_mask = (self.vehicles[:, :, 2:3] < 1)
        capacity_mask = capacity_mask.expand(-1, -1, self.nodes_count)
        capacity_mask[:, :, 0] = False
        
        # Combine all masks
        self.mask = (
            self.served.unsqueeze(1).expand(-1, self.veh_count, -1) |
            spoilage_mask |
            capacity_mask |
            self.veh_done.unsqueeze(-1).expand(-1, -1, self.nodes_count) |
            self.infeasible_nodes.unsqueeze(1).expand(-1, self.veh_count, -1)
        )
        self.mask[:, :, 0] = False
        
        # Update current vehicle mask
        self.cur_veh_mask = self.mask.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))
        
    def _update_cur_veh(self):
       avail = self.vehicles[:, :, 3].clone()
       avail[self.veh_done] = float('inf')
       self.cur_veh_idx = avail.argmin(1, keepdim=True)
       self.cur_veh = self.vehicles.gather(1,
           self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE))
       self.cur_veh_mask = self.mask.gather(1,
           self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))
       

    def reset(self):
       # Initialize vehicles [batch, veh_count, VEH_STATE_SIZE]
       self.vehicles = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.VEH_STATE_SIZE))
       self.vehicles[:, :, :2] = self.nodes[:, 0:1, :2]  # Position x,y
       self.vehicles[:, :, 2] = self.veh_capa           # Capacity
       self.vehicles[:, :, 3] = 0                       # Time


       # Calculate initial feasibility mask for nodes that can never be served
       depot_pos = self.nodes[:, 0:1, :2]  # [batch, 1, 2]
       node_pos = self.nodes[:, :, :2]     # [batch, nodes, 2]
       
       # Calculate round trip times from depot
       to_node_dist = (node_pos - depot_pos).norm(dim=-1)  # [batch, nodes]
       round_trip_time = 2 * (to_node_dist / self.veh_speed)  # Time for depot->node->depot
       
       # Mask nodes where round trip exceeds spoilage time
       self.infeasible_nodes = round_trip_time > self.nodes[:, :, 3]  # [batch, nodes]



       self.veh_done = self.nodes.new_zeros((self.minibatch_size, self.veh_count), dtype=torch.bool)
       self.done = False
       self.new_customers = True
       self.cust_mask = self.init_cust_mask
       self.served = self.nodes.new_zeros((self.minibatch_size, self.nodes_count), dtype=torch.bool)
       
       self.mask = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.nodes_count), dtype=torch.bool)
       if self.cust_mask is not None:
           self.mask = self.cust_mask[:, None, :].repeat(1, self.veh_count, 1)

       
       # Initialize urgent deadlines for vehicles as infinity
       self.urgent_deadlines = torch.full((self.minibatch_size, self.veh_count), float('inf'), 
                                        device=self.nodes.device)
      
           
       self.cur_veh_idx = self.nodes.new_zeros((self.minibatch_size, 1), dtype=torch.int64)
       self.cur_veh = self.vehicles.gather(1,
           self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE))



       self.cur_veh_mask = self.mask.gather(1,
           self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))
       
       self.vehicle_routes = self.nodes.new_zeros(
           (self.minibatch_size, self.nodes_count),
           dtype=torch.long
       ) - 1
       
       self.node_lateness_count = torch.zeros((self.minibatch_size, 1), device=self.nodes.device)
       self.late_nodes = torch.zeros((self.minibatch_size, self.nodes_count), dtype=torch.bool, device=self.nodes.device)


       

    def step(self, cust_idx):
        dest = self.nodes.gather(1, 
            cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE))

        dist, lateness, ontime_pickup = self._update_vehicles(dest, cust_idx)

        # Calculate immediate reward
        reward = (
            -self.dist_penalty_coef * dist
            - self.spoilage_penalty * lateness
            + self.pickup_bonus_coef * ontime_pickup
        )
        
        remaining_capacity = self.vehicles[:, :, 2]
        idle_vehicles_mask = (remaining_capacity == self.veh_capa).float()
        idle_penalty = -self.idle_penalty_coef * idle_vehicles_mask.sum(dim=1, keepdim=True)
        reward = reward + idle_penalty
        
        self._update_done(cust_idx)
        self._update_mask(cust_idx)
        self._update_cur_veh()
        
        if self.done:
            vehicle_times = self.vehicles[:, :, 3]
            arrival_times = self.nodes.new_zeros((self.minibatch_size, self.nodes_count))
            
            for v in range(self.veh_count):
                vehicle_mask = (self.vehicle_routes == v)
                arrival_times.masked_scatter_(
                    vehicle_mask,
                    vehicle_times[:, v].unsqueeze(1).expand(-1, vehicle_mask.size(1))
                )
            
            deadlines = self.nodes[:, :, 3]
            
            early_delivery = torch.clamp(deadlines - arrival_times, min=0)
            early_delivery = early_delivery * self.served.float()
            early_reward = self.early_reward * early_delivery.sum(dim=1, keepdim=True)
            
            late_delivery = torch.clamp(arrival_times - deadlines, min=0)
            late_delivery = late_delivery * self.served.float()
            late_penalty = self.additional_late_penalty * late_delivery.sum(dim=1, keepdim=True)
            
            if self.init_cust_mask is not None:
                self.served += self.init_cust_mask
            unserved = (self.served ^ True).float().sum(-1, keepdim=True) - 1
            unserved_penalty = -self.unserved_penalty * unserved
            
            reward = reward + early_reward - late_penalty + unserved_penalty
        
        self.vehicle_routes.scatter_(1,
            cust_idx,
            self.cur_veh_idx.expand(-1, cust_idx.size(1))
        )
        
        return reward


    def get_state(self):
       return None

    def state_dict(self, dest_dict=None):
       if dest_dict is None:
           dest_dict = {
               "vehicles": self.vehicles,
               "veh_done": self.veh_done,
               "served": self.served,
               "mask": self.mask,
               "cur_veh_idx": self.cur_veh_idx,
               "vehicle_routes": self.vehicle_routes
           }
       else:
           dest_dict["vehicles"].copy_(self.vehicles)
           dest_dict["veh_done"].copy_(self.veh_done)
           dest_dict["served"].copy_(self.served)
           dest_dict["mask"].copy_(self.mask)
           dest_dict["cur_veh_idx"].copy_(self.cur_veh_idx)
           dest_dict["vehicle_routes"].copy_(self.vehicle_routes)
       return dest_dict

    def load_state_dict(self, state_dict):
       self.vehicles.copy_(state_dict["vehicles"])
       self.veh_done.copy_(state_dict["veh_done"])
       self.served.copy_(state_dict["served"])
       self.mask.copy_(state_dict["mask"])
       self.cur_veh_idx.copy_(state_dict["cur_veh_idx"])
       self.vehicle_routes.copy_(state_dict["vehicle_routes"])
       self.cur_veh = self.vehicles.gather(1,
           self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE))
       self.cur_veh_mask = self.mask.gather(1,
           self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))