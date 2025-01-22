import torch
import copy

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
             dist_penalty_coef=1.0, pickup_bonus_coef=5.0, idle_penalty_coef=10.0,
             additional_late_penalty=10.0, capacity_usage_coef=0.4):
    # Existing initialization code...
       self.dist_penalty_coef = dist_penalty_coef
       self.pickup_bonus_coef = pickup_bonus_coef
       self.idle_penalty_coef = idle_penalty_coef
       self.additional_late_penalty = additional_late_penalty
       self.capacity_usage_coef = capacity_usage_coef
       super().__init__()
       self.veh_count = data.veh_count
       self.veh_capa = data.veh_capa
       self.veh_speed = data.veh_speed
       self.nodes = data.nodes if nodes is None else nodes
       self.init_cust_mask = data.cust_mask if cust_mask is None else cust_mask
       self.minibatch_size, self.nodes_count, _ = self.nodes.size()
       
       self.spoilage_penalty = spoilage_penalty
       self.early_reward = early_reward
       self.unserved_penalty = unserved_penalty
       if torch.cuda.device_count() > 1:
           self = nn.DataParallel(self)
            
    def _compute_travel_metrics(self, pos1, pos2):
        """Compute distances and travel times between positions"""
        distances = torch.cdist(pos1, pos2)  # [batch, src, dest]
        travel_times = distances / self.veh_speed
        return distances, travel_times   

       
    def _update_vehicles(self, dest, cust_idx):
        # Compute travel metrics in parallel
        dist = torch.pairwise_distance(self.cur_veh[:, 0, :2], dest[:, 0, :2], keepdim=True)
        travel_time = dist / self.veh_speed
        arrival_time = self.cur_veh[:, :, 3] + travel_time
        
        # Parallel depot distance computation
        to_depot_dist = torch.pairwise_distance(dest[:, 0, :2], self.nodes[:, 0, :2], keepdim=True)
        to_depot_time = to_depot_dist / self.veh_speed
        latest_arrival = dest[:, :, 3] - to_depot_time

        # Vectorized lateness check
        lateness = (arrival_time > latest_arrival).float()
        ontime_pickup = (~(arrival_time > latest_arrival).bool() & (cust_idx != 0)).float()

        # Update vehicle state
        self.cur_veh[:, :, :2] = dest[:, :, :2]
        self.cur_veh[:, :, 2] -= 1.0
        self.cur_veh[:, :, 3] += travel_time

        if self.cur_veh[:, :, 2].min() <= 0:
            self.cur_veh[:, :, :2] = self.nodes[:, 0:1, :2]
            self.cur_veh[:, :, 3] += to_depot_time
            self.vehicles = self.vehicles.scatter(1,
                self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE),
                self.cur_veh)

        return dist, lateness, ontime_pickup
        

        
      
   
    def _update_mask(self, cust_idx):
        self.new_customers = False
        # Mark served customers
        self.served.scatter_(1, cust_idx, cust_idx > 0)
        
        # Update urgent deadline
        # if cust_idx.any():
        #     new_deadlines = self.nodes.gather(1, cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE))[:, :, 3]
        #     self.urgent_deadlines.scatter_(1, self.cur_veh_idx, 
        #         torch.min(self.urgent_deadlines.gather(1, self.cur_veh_idx), new_deadlines))

        # Capacity mask - Fix the dimension issue
        capacity_mask = self.vehicles[:, :, 2:3] < 1
        capacity_mask = capacity_mask.expand(-1, -1, self.nodes_count)
        capacity_mask[:, :, 0] = False



        # Time feasibility calculations
        depot_pos = self.nodes[:, 0:1, :2]
        node_pos = self.nodes[:, :, :2]
        current_time = self.cur_veh[:, :, 3]
        
        # Calculate travel times
        to_node_dist = (node_pos.unsqueeze(1) - self.cur_veh[:, :, None, :2]).norm(dim=-1)
        to_node_time = to_node_dist / self.veh_speed
        
        to_depot_dist = (node_pos - depot_pos).norm(dim=-1)
        to_depot_time = to_depot_dist / self.veh_speed
        
        # Total delivery time
        #total_time = current_time.unsqueeze(-1) + to_node_time + to_depot_time.unsqueeze(1)
        
        # Time mask
        #time_mask = total_time > self.urgent_deadlines.unsqueeze(-1)

        # Combine masks with proper dimensions
        self.mask = (
            self.mask |                                                         # [batch, veh_count, nodes_count]
            self.served.unsqueeze(1).expand(-1, self.veh_count, -1) |         # [batch, veh_count, nodes_count]
            capacity_mask |                                                     # [batch, veh_count, nodes_count]
            self.veh_done.unsqueeze(-1).expand(-1, -1, self.nodes_count) |    # [batch, veh_count, nodes_count]
            self.infeasible_nodes.unsqueeze(1).expand(-1, self.veh_count, -1) # [batch, veh_count, nodes_count]
        )
        # Never mask the depot
        self.mask[:, :, 0] = 0  # Depot always available

        # Update current vehicle mask
        self.cur_veh_mask = self.mask.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))
        
        
    def _update_cur_veh(self):
        # Find the vehicle with the minimum available time and capacity
        avail = self.vehicles[:, :, 3].clone()
        avail[self.veh_done] = float('inf')
        avail[self.vehicles[:, :, 2] <1] = float('inf')
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
        self.cust_mask = self.init_cust_mask
        self.new_customers = True
        self.served = self.nodes.new_zeros((self.minibatch_size, self.nodes_count), dtype=torch.bool)

        self.arrival_times = self.nodes.new_zeros((self.minibatch_size, self.nodes_count))

        self.mask = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.nodes_count), dtype = torch.bool) \
                if self.cust_mask is None \
                else self.cust_mask[:,None,:].repeat(1, self.veh_count, 1)
        
        #    self.mask = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.nodes_count), dtype=torch.bool)
        #    if self.cust_mask is not None:
        #        self.mask = self.cust_mask[:, None, :].repeat(1, self.veh_count, 1)

        
        # Initialize urgent deadlines for vehicles as infinity
        # self.urgent_deadlines = torch.full((self.minibatch_size, self.veh_count), float('inf'), 
        #                                     device=self.nodes.device)
        
            
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



    def clone(self):
        """Create a deep copy of the environment state"""
        cloned = copy.deepcopy(self)
        # Reset flags but keep state
        cloned.done = False
        cloned.new_customers = False
        return cloned


    def step(self, cust_idx):
        dest = self.nodes.gather(1, 
            cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE))
    
        dist, lateness, ontime_pickup = self._update_vehicles(dest, cust_idx)

        # Calculate immediate rewards
        dist_penalty = -self.dist_penalty_coef * dist
        late_penalty = -self.spoilage_penalty * lateness
        pickup_bonus = self.pickup_bonus_coef * ontime_pickup

        # Calculate idle penalty
        remaining_capacity = self.vehicles[:, :, 2]
        idle_vehicles_mask = (remaining_capacity == self.veh_capa).float()
        idle_penalty = -self.idle_penalty_coef * idle_vehicles_mask.sum(dim=1, keepdim=True)

        rewards = dist_penalty + late_penalty + idle_penalty + pickup_bonus
        
        
        self._update_done(cust_idx)
        self._update_mask(cust_idx)
        self._update_cur_veh()

        


        if self.done:
        # Calculate final arrival times and penalties
            vehicle_times = self.vehicles[:, :, 3]
            arrival_times = self.nodes.new_zeros((self.minibatch_size, self.nodes_count))
            
            # Update arrival times for all nodes that were served
            vehicle_routes_expanded = self.vehicle_routes.unsqueeze(-1)  # [batch, nodes, 1]
            vehicle_times_expanded = vehicle_times.unsqueeze(1)  # [batch, 1, veh_count]
            route_mask = (vehicle_routes_expanded == torch.arange(self.veh_count, device=self.nodes.device).view(1, 1, -1))
            arrival_times = (route_mask * vehicle_times_expanded).sum(dim=-1)  # [batch, nodes]

            # Calculate lateness penalties
            deadlines = self.nodes[:, :, 3]
            depot_late = (arrival_times > deadlines)
            depot_late_count = depot_late.float().sum(dim=1, keepdim=True)
            additional_late_count = torch.clamp(depot_late_count - self.node_lateness_count, min=0)
            
            additional_lateness_penalty = -self.additional_late_penalty  * additional_late_count
            continuous_lateness_penalty = -self.spoilage_penalty * torch.clamp(
                arrival_times - deadlines, min=0).sum(dim=1, keepdim=True)

            # Update served nodes and calculate unserved penalty
            if self.init_cust_mask is not None:
                self.served += self.init_cust_mask
            unserved = ((self.served ^ True) & ~self.infeasible_nodes).float().sum(-1, keepdim=True) - 1
            
            # Calculate route balance penalties
            # route_lengths = torch.zeros((self.minibatch_size, self.veh_count), device=self.nodes.device)
            # for v in range(self.veh_count):
            #     route_lengths[:, v] = (self.vehicle_routes == v).sum(dim=1)
            # route_variance = route_lengths.var(dim=1, keepdim=True)
            
            capacity_usage = remaining_capacity / self.veh_capa
            
            # Add final penalties
            rewards += (additional_lateness_penalty 
                    + continuous_lateness_penalty  
                    - self.unserved_penalty * unserved
                    -self.capacity_usage_coef * capacity_usage.mean(dim=1, keepdim=True))

        # Update route tracking
        self.vehicle_routes.scatter_(1,
            cust_idx,
            self.cur_veh_idx.expand(-1, cust_idx.size(1)))

        return rewards

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