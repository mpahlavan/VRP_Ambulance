
import torch
import logging
import os
from datetime import datetime


class PVRPLogger:
    def __init__(self, log_file="+pvrp_run.log"):
        self.logger = logging.getLogger('PVRP')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        
        formatter = logging.Formatter('%(asctime)s | %(message)s', 
                                   datefmt='%Y-%m-%d %H:%M:%S')
        
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def log_section(self, title):
        self.logger.info(f"\n{'='*20} {title} {'='*20}\n")


class PVRP_Environment:
    VEH_STATE_SIZE = 4
    CUST_FEAT_SIZE = 4

    def __init__(self, data, nodes=None, cust_mask=None,
                 spoilage_penalty=10.0, early_reward=5, unserved_penalty=10.0,
                 dist_penalty_coef=1.0, pickup_bonus_coef=5.0, idle_penalty_coef=20.0,
                 additional_late_penalty=100.0, capacity_usage_coef=0.4, log_file=None):
        
        
        if log_file is None:
            # Create logs directory if it doesn't exist
            os.makedirs('++logs', exist_ok=True)
            # Generate unique log file name based on timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'++logs/pvrp_run_{timestamp}.log'
        
        self.logger = PVRPLogger(log_file=log_file).logger

        # Store parameters
        self.veh_count = data.veh_count
        self.veh_capa = data.veh_capa
        self.veh_speed = data.veh_speed
        self.nodes = data.nodes if nodes is None else nodes
        self.init_cust_mask = data.cust_mask if cust_mask is None else cust_mask
        self.minibatch_size, self.nodes_count, _ = self.nodes.size()

        # Reward parameters
        self.spoilage_penalty = spoilage_penalty
        self.early_reward = early_reward
        self.unserved_penalty = unserved_penalty
        self.dist_penalty_coef = dist_penalty_coef
        self.pickup_bonus_coef = pickup_bonus_coef
        self.idle_penalty_coef = idle_penalty_coef
        self.additional_late_penalty = additional_late_penalty
        self.capacity_usage_coef = capacity_usage_coef
        
        # Log initialization
        self.logger.info("Environment Configuration:")
        self.logger.info(f"Vehicles: {self.veh_count} | Capacity: {self.veh_capa} | Speed: {self.veh_speed}")
        self.logger.info(f"Reward Parameters:")
        self.logger.info(f"- Spoilage Penalty: {self.spoilage_penalty}")
        self.logger.info(f"- Early Reward: {self.early_reward}")
        self.logger.info(f"- Unserved Penalty: {self.unserved_penalty}")
        self.logger.info(f"- Other Coefficients: dist={self.dist_penalty_coef}, pickup={self.pickup_bonus_coef}")

     



  



       
  
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

        
        # # Log updates
        # path_info = []
        # for b in range(self.minibatch_size):
        #     if cust_idx[b].item() != 0:  # For customer nodes
        #         path_info.append(
        #             f"PATH | B{b:03d}-V{self.cur_veh_idx[b].item()}: "
        #             f"N{cust_idx[b].item():02d} | "
        #             f"Arrival={arrival_time[b].item():.1f} | "
        #             f"Latest={latest_arrival[b].item():.1f} | "
        #             f"{'LATE' if lateness[b].item() > 0 else 'On-time'} | "
        #             f"Cap={self.cur_veh[b, 0, 2].item():.1f}"
        #         )
        #     else:  # For depot
        #         path_info.append(
        #             f"PATH | B{b:03d}-V{self.cur_veh_idx[b].item()}: "
        #             f"Depot | "
        #             f"Arrival={arrival_time[b].item():.1f} | "
        #             f"Cap={self.cur_veh[b, 0, 2].item():.1f}"
        #         )
                
        # if path_info:
        #     self.logger.info('\n'.join(path_info))
        
        
        return dist, lateness, ontime_pickup
    
    
    def _update_done(self, cust_idx):
        self.veh_done.scatter_(1, self.cur_veh_idx, cust_idx == 0)
        self.done = bool(self.veh_done.all())
        self.logger.debug(f"Done status: {self.done}")


        
    def _update_mask(self, cust_idx):
        """Update mask with detailed logging of node availability for each vehicle"""
        self.new_customers = False
        
        # Update served nodes
        prev_served = self.served.clone()  # Store previous state for logging
        self.served.scatter_(1, cust_idx, cust_idx > 0)
        
        # Log served node update
        # for b in range(self.minibatch_size):
        #     if cust_idx[b].item() != 0:  # Skip depot
        #         self.logger.info(f"MASK | B{b:03d} | Node {cust_idx[b].item():02d} marked as served")

        # Create capacity mask
        capacity_mask = torch.zeros_like(self.mask)
        scatter_indices = self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count)
        capacity_condition = (self.cur_veh[:, :, 2] < 1)
        capacity_condition = capacity_condition[:, :, None].expand(-1, -1, self.nodes_count)
        capacity_mask.scatter_(1, scatter_indices, capacity_condition)
        capacity_mask[:, :, 0] = 0  # Reset depot mask

    
        # Combine all masks
        self.mask = (
            self.mask |
            self.served.unsqueeze(1).expand(-1, self.veh_count, -1) |  # Served nodes
            capacity_mask |  # Capacity constraints
            self.veh_done.unsqueeze(-1).expand(-1, -1, self.nodes_count)   # Done vehicles
        )
        self.mask[:, :, 0] = 0  # Ensure depot is always available

    
        # Update current vehicle mask
      
    def _update_cur_veh(self):
        avail = self.vehicles[:, :, 3].clone()
        avail[self.veh_done] = float('inf')
        avail[self.vehicles[:, :, 2] <1] = float('inf')
        self.cur_veh_idx = avail.argmin(1, keepdim=True)

        self.cur_veh = self.vehicles.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE))
        self.cur_veh_mask = self.mask.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))
        
        

    def reset(self):
        self.vehicles = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.VEH_STATE_SIZE))
        self.vehicles[:, :, :2] = self.nodes[:, 0:1, :2]
        self.vehicles[:, :, 2] = self.veh_capa
     

        self.veh_done = self.nodes.new_zeros((self.minibatch_size, self.veh_count), dtype=torch.bool)
        self.done = False
        
        self.cust_mask = self.init_cust_mask
        self.new_customers = True
        self.served = self.nodes.new_zeros((self.minibatch_size, self.nodes_count), dtype=torch.bool)
        
        self.mask = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.nodes_count), dtype = torch.bool) \
                if self.cust_mask is None else self.cust_mask[:,None,:].repeat(1, self.veh_count, 1)

        self.urgent_deadlines = torch.full(
            (self.minibatch_size, self.veh_count),
            float('inf'),
            device=self.nodes.device
        )
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
        self.node_lateness_count = torch.zeros((self.minibatch_size, 1), device=self.nodes.device)
        self.depot_lateness_count = torch.zeros((self.minibatch_size, 1), device=self.nodes.device)
        self.late_nodes = torch.zeros((self.minibatch_size, self.nodes_count), dtype=torch.bool, device=self.nodes.device)
        self.node_depot_times = self.nodes.new_full((self.minibatch_size, self.nodes_count), float('inf'))
        
        
        
        return None


    def step(self, cust_idx):
        dest = self.nodes.gather(1, 
            cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE))

        dist, lateness, ontime_pickup = self._update_vehicles(dest, cust_idx)
        self._update_done(cust_idx)
        self._update_mask(cust_idx)
        self._update_cur_veh()

        # Immediate rewards
        distance_penalty = -self.dist_penalty_coef * dist
        spoilage_penalty = -self.spoilage_penalty * lateness
        pickup_bonus = self.pickup_bonus_coef * ontime_pickup
        reward = spoilage_penalty + pickup_bonus + distance_penalty

        
        
        """
        Calculate final rewards at episode end with proper deadline handling.
        
        Tensor dimensions:
        - arrival_time: [batch_size, veh_count]
        - deadlines: [batch_size, nodes_count, 4]
        - urgent_deadline: [batch_size, veh_count]
        - deadline_gap: [batch_size, veh_count]
        """
        if self.done:
            # Get final vehicle arrival times
            # arrival_time = self.vehicles[:, :, 3]  # [batch_size, veh_count]
            
            
            
            # # Calculate urgent deadlines for each vehicle
            # urgent_deadline = torch.full(
            #     (self.minibatch_size, self.veh_count), 
            #     float('inf'), 
            #     device=self.nodes.device
            # )
            # spoilage_times = self.nodes[:, :, 3]  # [batch_size, nodes_count]
            
            # # Find minimum spoilage time for goods carried by each vehicle
            # for v in range(self.veh_count):
            #     mask_v = (self.vehicle_routes == v)  # [batch_size, nodes_count]
            #     for b in range(self.minibatch_size):
            #         # Get indices of nodes served by vehicle v (excluding depot)
            #         indices = torch.nonzero(
            #             mask_v[b] & (torch.arange(self.nodes_count, device=self.nodes.device) != 0),
            #             as_tuple=False
            #         ).squeeze(-1)
                    
            #         # Update urgent deadline if vehicle served any nodes
            #         if indices.numel() > 0:
            #             urgent_deadline[b, v] = spoilage_times[b, indices].min()

    

            
            # # Calculate gap between urgent deadlines and vehicle arrival times
            # deadline_gap = urgent_deadline - arrival_time  # [batch_size, veh_count]
            # deadline_gap_total =  deadline_gap.sum(dim=1, keepdim=True)  # [batch_size, 1]
            # deadline_penalty= deadline_gap_total * self.additional_late_penalty
            # # Log detailed deadline information
            # for b in range(self.minibatch_size):
            #     vehicle_info = []
            #     for v in range(self.veh_count):
            #         if urgent_deadline[b, v] < float('inf'):
            #             vehicle_info.append(
            #                 f"V{v}: arrival={arrival_time[b,v]:.1f}, "
            #                 f"deadline={urgent_deadline[b,v]:.1f}, "
            #                 f"gap={deadline_gap[b,v]:.1f}"
            #             )
            #     if vehicle_info:
            #         self.logger.info(f"DEADLINE | B{b:03d} | " + " | ".join(vehicle_info))
            if self.init_cust_mask is not None:
                self.served += self.init_cust_mask
                    
                    # Unserved customers penalty
                unserved = ((~self.served & ~self.infeasible_nodes).float().sum(-1, keepdim=True))* unserved_penalty  # [batch, 1]
            # Update final reward
            reward = reward + unserved

        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, device=self.nodes.device)
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        

       
                   
                    
                  
        # # Log final statistics
        #         self.logger.info("-" * 50)
        #         for b in range(self.minibatch_size):
        #             self.logger.info(
        #                 f"[FINAL] B{b:03d} | "
        #                 f"Served: {self.served[b].sum().item():02d} | "
        #                 f"Unserved: {unserved_penalty[b].item():.1f} | "
        #                 f"DeadlineGap: {deadline_gap_total[b].item():.1f} | "
        #                 f"Total: {reward[b].item():.1f}"
        #             )
        #         self.logger.info("-" * 50)
        
        return reward

   



    def log_state(self):
        """Log complete environment state"""
        self.logger.info("Current Environment State:")
        self.logger.info(f"Vehicles:\n{self.vehicles.cpu().numpy()}")
        self.logger.info(f"Served nodes:\n{self.served.cpu().numpy()}")
        self.logger.info(f"Mask:\n{self.mask.cpu().numpy()}")
        self.logger.info(f"Current vehicle indices: {self.cur_veh_idx.cpu().numpy().tolist()}")



    
    def state_dict(self, dest_dict=None):
        if dest_dict is None:
            return {
                "vehicles": self.vehicles,
                "veh_done": self.veh_done,
                "served": self.served,
                "mask": self.mask,
                "cur_veh_idx": self.cur_veh_idx,
                "vehicle_routes": self.vehicle_routes
            }
        else:
            dest_dict.update({
                "vehicles": self.vehicles.clone(),
                "veh_done": self.veh_done.clone(),
                "served": self.served.clone(),
                "mask": self.mask.clone(),
                "cur_veh_idx": self.cur_veh_idx.clone(),
                "vehicle_routes": self.vehicle_routes.clone()
            })
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
