import torch

class PVRP_Environment:
   
   VEH_STATE_SIZE = 4  # position(2), capacity(1), time(1)
   CUST_FEAT_SIZE = 4  # x,y, demand(1), spoilage_time

   def __init__(self, data, nodes=None, cust_mask=None,
                spoilage_penalty=2.0, early_reward=0.5, unserved_penalty=1.0):
       self.veh_count = data.veh_count
       self.veh_capa = data.veh_capa
       self.veh_speed = data.veh_speed
       self.nodes = data.nodes if nodes is None else nodes
       self.init_cust_mask = data.cust_mask if cust_mask is None else cust_mask
       self.minibatch_size, self.nodes_count, _ = self.nodes.size()
       
       self.spoilage_penalty = spoilage_penalty
       self.early_reward = early_reward
       self.unserved_penalty = unserved_penalty
       self.new_customers = True

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

       # Initialize vehicle states and tracking
       self.veh_done = self.nodes.new_zeros((self.minibatch_size, self.veh_count), dtype=torch.bool)
       self.done = False
       self.cust_mask = self.init_cust_mask
       self.served = self.nodes.new_zeros((self.minibatch_size, self.nodes_count), dtype=torch.bool)
       
       # Initialize action masks
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

   def _update_done(self, cust_idx):
       self.veh_done.scatter_(1, self.cur_veh_idx, cust_idx == 0)
       self.done = bool(self.veh_done.all())

   def _update_mask(self, cust_idx):
        self.served.scatter_(1, cust_idx, cust_idx > 0)
        
        # Update urgent deadline when picking up new goods
        if cust_idx.any():  # If not returning to depot
            new_deadlines = self.nodes.gather(1, cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE))[:, :, 3]
            self.urgent_deadlines.scatter_(1, self.cur_veh_idx, 
                torch.min(self.urgent_deadlines.gather(1, self.cur_veh_idx), new_deadlines))

        # Capacity constraint
        # Capacity constraint - Fix the dimension issue
        capacity_mask = (self.cur_veh[:, :, 2] <= 0).unsqueeze(-1)  # [batch, 1, 1]
        capacity_mask = capacity_mask.expand(-1, -1, self.nodes_count)  # [batch, 1, nodes]

        '''
        overload = torch.zeros_like(self.mask).scatter_(
            1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count),
            self.cur_veh[:, :, None, 2] < 1)
        '''

        # Time feasibility calculations
        depot_pos = self.nodes[:, 0:1, :2]  
        node_pos = self.nodes[:, :, :2]     
        current_time = self.cur_veh[:, :, 3] 
        
        # Calculate travel times for current position to node to depot
        to_node_dist = (node_pos.unsqueeze(1) - self.cur_veh[:, :, None, :2]).norm(dim=-1)
        to_node_time = to_node_dist / self.veh_speed
        
        to_depot_dist = (node_pos - depot_pos).norm(dim=-1)
        to_depot_time = to_depot_dist / self.veh_speed
        
        # Total delivery time if we visit each node
        total_time = current_time.unsqueeze(-1) + to_node_time + to_depot_time.unsqueeze(1)
        
        # Check only against urgent deadline of goods in vehicle
        time_mask = total_time > self.urgent_deadlines.unsqueeze(-1)
        
        # Combine masks
        self.mask = (self.served.unsqueeze(1) |
                    capacity_mask |
                    time_mask |
                    self.veh_done.unsqueeze(-1) |
                    self.infeasible_nodes.unsqueeze(1))  # Include initially infeasible nodes
        self.mask[:, :, 0] = 0  # Depot always available

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

   def step(self, cust_idx):
       # Get destination info
       dest = self.nodes.gather(1, 
           cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE))
       
       # Update vehicle state and get travel distance
       dist = torch.pairwise_distance(self.cur_veh[:, 0, :2], dest[:, 0, :2], keepdim=True)
       travel_time = dist / self.veh_speed

       self.cur_veh[:, :, :2] = dest[:, :, :2]  # Position x,y
       self.cur_veh[:, :, 2] -= 1.0             # Capacity
       self.cur_veh[:, :, 3] += travel_time     # Time

       self.vehicles = self.vehicles.scatter(1,
           self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE),
           self.cur_veh)
       
       # Update state 
       self._update_done(cust_idx)
       self._update_mask(cust_idx)
       self._update_cur_veh()
       
       # Calculate rewards
       rewards = -dist
       
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
           late_penalty = self.spoilage_penalty * late_delivery.sum(dim=1, keepdim=True)
           
           # Don't penalize initially infeasible nodes
           if self.init_cust_mask is not None:
               self.served += self.init_cust_mask
           unserved = ((self.served ^ True) & ~self.infeasible_nodes).float().sum(-1, keepdim=True) - 1
           rewards += early_reward - late_penalty - self.unserved_penalty * unserved
       
       # Update route tracking
       self.vehicle_routes.scatter_(1,
           cust_idx,
           self.cur_veh_idx.expand(-1, cust_idx.size(1))
       )
       
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
               "vehicle_routes": self.vehicle_routes,
               "urgent_deadlines": self.urgent_deadlines,
               "infeasible_nodes": self.infeasible_nodes
           }
       else:
           dest_dict["vehicles"].copy_(self.vehicles)
           dest_dict["veh_done"].copy_(self.veh_done)
           dest_dict["served"].copy_(self.served)
           dest_dict["mask"].copy_(self.mask)
           dest_dict["cur_veh_idx"].copy_(self.cur_veh_idx)
           dest_dict["vehicle_routes"].copy_(self.vehicle_routes)
           dest_dict["urgent_deadlines"].copy_(self.urgent_deadlines)
           dest_dict["infeasible_nodes"].copy_(self.infeasible_nodes)
       return dest_dict

   def load_state_dict(self, state_dict):
       self.vehicles.copy_(state_dict["vehicles"])
       self.veh_done.copy_(state_dict["veh_done"])
       self.served.copy_(state_dict["served"])
       self.mask.copy_(state_dict["mask"])
       self.cur_veh_idx.copy_(state_dict["cur_veh_idx"])
       self.vehicle_routes.copy_(state_dict["vehicle_routes"])
       self.urgent_deadlines.copy_(state_dict["urgent_deadlines"])
       self.infeasible_nodes.copy_(state_dict["infeasible_nodes"])
       self.cur_veh = self.vehicles.gather(1,
           self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE))
       self.cur_veh_mask = self.mask.gather(1,
           self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))