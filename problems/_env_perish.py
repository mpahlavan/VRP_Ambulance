import torch
import os
from datetime import datetime

class PVRP_Custom_Logger:
    """
    Custom format logger for PVRP Environment
    Outputs logs in the requested specific format
    """
    def __init__(self, log_dir=None, file_name=None):
        self.log_dir = log_dir or f"logs/pvrp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.file_name = file_name or "pvrp_log.txt"
        self.log_path = os.path.join(self.log_dir, self.file_name)
        
        # Create directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize log file
        with open(self.log_path, 'w') as f:
            f.write(f"# PVRP Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Format: route, mask, final statistics\n\n")
    
    def info(self, message):
        """
        General logging method that writes any message to the log file
        """
        with open(self.log_path, 'a') as f:
            f.write(f"{message}\n")
    def log_route(self, batch_idx, veh_idx, node_idx, arrival_time, latest_time, capacity):
        """
        Log a vehicle route step
        
        Example:
        route| B003-V0: N07 | Arrival=0.4 | Latest=0.6 | On-time | Cap=4.0
        """
        status = "On-time" if arrival_time <= latest_time else "Late"
        
        with open(self.log_path, 'a') as f:
            f.write(f"route| B{batch_idx:03d}-V{veh_idx}: N{node_idx:02d} | ")
            f.write(f"Arrival={arrival_time:.1f} | Latest={latest_time:.1f} | {status} | Cap={capacity:.1f}\n")
    
    def log_mask(self, batch_idx, veh_idx, visible_nodes):
        """
        Log which nodes are visible/feasible for a vehicle
        
        Example:
        | MASK | B000 | V0 sees nodes: [1, 2, 3, 4, 5, 6, 7, 9, 10]
        """
        # Format nodes as a comma-separated list
        nodes_str = ", ".join([f"{n}" for n in visible_nodes])
        
        with open(self.log_path, 'a') as f:
            f.write(f"| MASK | B{batch_idx:03d} | V{veh_idx} sees nodes: [{nodes_str}]\n")
    
    def log_final_stats(self, batch_idx, served_count, unserved_penalty, total_cost):
        """
        Log final statistics for a batch
        
        Example:
        [FINAL] B000 | Served: 01 | Unserved: -90.0 | Total: -100.8
        """
        with open(self.log_path, 'a') as f:
            f.write(f"[FINAL] B{batch_idx:03d} | Served: {served_count:02d} | ")
            f.write(f"Unserved: {unserved_penalty:.1f} | Total: {total_cost:.1f}\n")
    
    def log_separator(self):
        """Add a separator line to the log"""
        with open(self.log_path, 'a') as f:
            f.write("\n")

# Modified PVRP Environment with custom format logging
class PVRP_Environment:
    VEH_STATE_SIZE = 4  # position(2), capacity(1), time(1)
    CUST_FEAT_SIZE = 4  # x,y, demand(1), spoilage_time

    def __init__(self, data, nodes=None, cust_mask=None,
                spoilage_penalty=1, unserved_penalty=1,pickup_bonus_coef=1,
                additional_late_penalty=1, capacity_usage_coef=0.0,dist_penalty_coef=0.05,idle_penalty_coef =10,success_bonus=10, logger=None):
        self.veh_count = data.veh_count
        self.veh_capa = data.veh_capa
        self.veh_speed = data.veh_speed
        self.nodes = data.nodes if nodes is None else nodes
        self.init_cust_mask = data.cust_mask if cust_mask is None else cust_mask
        self.minibatch_size, self.nodes_count, _ = self.nodes.size()
        
        self.spoilage_penalty = spoilage_penalty
        self.unserved_penalty = unserved_penalty
        self.dist_penalty_coef = dist_penalty_coef
        self.pickup_bonus_coef = pickup_bonus_coef
        self.additional_late_penalty = additional_late_penalty
        self.capacity_usage_coef = capacity_usage_coef
        # self.early_reward = early_reward
        self.idle_penalty_coef = idle_penalty_coef
        self.last_reward = torch.zeros((self.minibatch_size, 1), device=self.nodes.device)
        self.success_bonus = success_bonus 
       
        
        # Initialize logger
        self.logger = logger or PVRP_Custom_Logger()
        
        # Track total costs per batch
        self.batch_costs = torch.zeros(self.minibatch_size)
        self.batch_served = torch.zeros(self.minibatch_size, dtype=torch.long)

        self.late_nodes = None
        self.node_lateness_count = 0

   
    
    def _update_vehicles(self, dest, cust_idx):
        """
        Update vehicle states after moving to destination
        
        Args:
            dest: Destination node features
            cust_idx: Customer indices
            
        Returns:
            dist: Travel distance
            lateness: Binary indicator of lateness
            ontime_pickup: Binary indicator of on-time pickup
        """
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
        
        # Calculate latest allowed arrival time (spoilage time minus time to depot)
        latest_arrival = dest[:, :, 3] - to_depot_time
        
        # Check if delivery was late
        lateness = (arrival_time > latest_arrival).float()
        not_late = ~(lateness.bool())
        is_customer = (cust_idx > 0).float()
        ontime_pickup = (not_late.float() * is_customer)
        
        # Track late nodes
        if cust_idx.any():
            self.late_nodes.scatter_(1, cust_idx, lateness.bool())
            self.node_lateness_count += lateness.sum().item()
        
        # Update vehicle routes for customer nodes
        for b in range(self.minibatch_size):
            node_idx = cust_idx[b, 0].item()
            if node_idx > 0:  # فقط برای گره‌های مشتری
                veh_idx = self.cur_veh_idx[b, 0].item()
                self.vehicle_routes[b, node_idx] = veh_idx
        
        # ----- CAPACITY TRACKING -----
        # Get current capacity before update
        old_capacity = self.cur_veh[:, :, 2].clone()
        
        # Update vehicle state
        self.cur_veh[:, :, :2] = dest[:, :, :2]  # Position x,y
        #TODO:capability checking
        #self.cur_veh[:, :, 2] -= 1.0             # Capacity
        self.cur_veh[:, :, 3] += travel_time     # Time
        

        # Update capacity only for customer visits (not depot)
        # Demand is 1 unit per customer
        customer_visit_mask = (cust_idx > 0).float()
        capacity_change = -1.0 * customer_visit_mask
        self.cur_veh[:, :, 2] += capacity_change  # Decrease capacity

        # Update visit count for capacity tracking
        for b in range(self.minibatch_size):
            if cust_idx[b, 0].item() > 0:  # Customer node
                v_idx = self.cur_veh_idx[b, 0].item()
                self.vehicle_visit_count[b, v_idx] += 1
        # ---------------------------
        # Update vehicles tensor
        self.vehicles = self.vehicles.scatter(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE),
            self.cur_veh)
        
        # Log vehicle paths and capacity
        path_info = []
        for b in range(self.minibatch_size):
            node_idx = cust_idx[b, 0].item()
            veh_idx = self.cur_veh_idx[b, 0].item()
            new_capacity = self.cur_veh[b, 0, 2].item()
            visit_count = self.vehicle_visit_count[b, veh_idx].item()
            
            if node_idx > 0:  # Customer node
                path_info.append(
                    f"PATH | B{b:03d}-V{veh_idx}: "
                    f"N{node_idx:02d} | "
                    f"Arrival={arrival_time[b, 0].item():.1f} | "
                    f"Latest={latest_arrival[b, 0].item():.1f} | "
                    f"{'LATE' if lateness[b, 0].item() > 0 else 'On-time'} | "
                    f"Cap: {old_capacity[b, 0].item():.1f}→{new_capacity:.1f} | "
                    f"Visits: {visit_count}"
                )
            else:  # Depot
                path_info.append(
                    f"PATH | B{b:03d}-V{veh_idx}: "
                    f"Depot | "
                    f"Arrival={arrival_time[b, 0].item():.1f} | "
                    f"Cap={new_capacity:.1f} | "
                    f"Visits: {visit_count}"
                )
        
        if path_info:
            self.logger.info('\n'.join(path_info))
        
        
        return dist, lateness, ontime_pickup
    
    def _update_done(self, cust_idx):
        self.veh_done.scatter_(1, self.cur_veh_idx, cust_idx == 0)
        self.done = bool(self.veh_done.all())
   
    def _update_mask(self, cust_idx):
        # Mark served customers
        self.served.scatter_(1, cust_idx, cust_idx > 0)
        
        # Update urgent deadline
        # if cust_idx.any():
        #     new_deadlines = self.nodes.gather(1, cust_idx[:, :, None].expand(-1, -1, self.CUST_FEAT_SIZE))[:, :, 3]
        #     self.urgent_deadlines.scatter_(1, self.cur_veh_idx, 
        #         torch.min(self.urgent_deadlines.gather(1, self.cur_veh_idx), new_deadlines))

         # ----- CAPACITY MASK FIX -----
        # Get remaining capacity for current vehicle
        # ----- CAPACITY MASK FIX -----
        # Get remaining capacity for current vehicle
        remaining_capacity = self.cur_veh[:, :, 2]  # [batch, 1]

        # Create capacity mask: vehicle can't visit nodes if capacity < 1
        capacity_mask = (remaining_capacity <= 0)  # [batch, 1]

        # Expand to mask all nodes if capacity constraint is violated
        capacity_mask = capacity_mask.unsqueeze(-1).expand(-1, -1, self.nodes_count)  # [batch, 1, nodes]

        # Create a full mask of zeros for all vehicles
        full_capacity_mask = torch.zeros_like(self.mask, dtype=torch.bool)  # [batch, veh_count, nodes]

        # Apply capacity mask only to the current vehicle
        for b in range(self.minibatch_size):
            v_idx = self.cur_veh_idx[b, 0].item()
            full_capacity_mask[b, v_idx] = capacity_mask[b, 0]

        # Enhanced debug logging for capacity
        for b in range(self.minibatch_size):
            v_idx = self.cur_veh_idx[b, 0].item()
            is_at_capacity = capacity_mask[b, 0, 0].item()  # Check if vehicle is at capacity
            affected_nodes = capacity_mask[b, 0].sum().item() if is_at_capacity else 0
            
            self.logger.info(f"CAPACITY | B{b:03d} | V{v_idx} | " +
                            f"Remaining={remaining_capacity[b, 0].item():.1f} | " +
                            f"At Capacity={is_at_capacity} | " +
                            f"Affected Nodes={affected_nodes} | " +
                            f"Other Vehicles Protected=True")
        # Combine masks with proper dimensions
        # Combine all masks
        self.mask = (
            self.mask |
            self.served.unsqueeze(1).expand(-1, self.veh_count, -1) |  # Served nodes
            full_capacity_mask |  # Capacity constraints
            self.veh_done.unsqueeze(-1).expand(-1, -1, self.nodes_count) |  # Done vehicles
            self.infeasible_nodes.unsqueeze(1)
        )
        self.mask[:, :, 0] = 0  # Depot always available

        # Update current vehicle mask
        self.cur_veh_mask = self.mask.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))
        
        # Log visible nodes for current vehicle
        for b in range(self.minibatch_size):
            veh_idx = self.cur_veh_idx[b, 0].item()
            visible_nodes = (~self.cur_veh_mask[b, 0]).nonzero().flatten().tolist()
            visible_str = ", ".join(str(n) for n in visible_nodes if n > 0)
            if visible_str:
                self.logger.info(f"| MASK | B{b:03d} | V{veh_idx} sees nodes: [{visible_str}]")
            else:
                self.logger.info(f"| MASK | B{b:03d} | V{veh_idx} sees no nodes (capacity={remaining_capacity[b, 0].item():.1f})")
    
    
    # def _update_cur_veh(self):
    #     """
    #     Select the next vehicle using round-robin assignment
    #     This ensures all vehicles get a chance to be utilized
    #     """
    #     batch_size = self.minibatch_size
        
    #     # Initialize next_vehicle_idx tensor if first call
    #     if not hasattr(self, 'next_vehicle_idx'):
    #         self.next_vehicle_idx = torch.zeros((batch_size,), dtype=torch.long, device=self.nodes.device)
        
    #     # Store current vehicle indices before updating
    #     self.cur_veh_idx = torch.zeros((batch_size, 1), dtype=torch.long, device=self.nodes.device)
        
    #     # Implement round-robin selection for each batch
    #     for b in range(batch_size):
    #         # Start from the next vehicle in sequence
    #         v_idx = self.next_vehicle_idx[b]
            
    #         # Find next available vehicle (not done)
    #         attempts = 0
    #         while self.veh_done[b, v_idx] and attempts < self.veh_count:
    #             v_idx = (v_idx + 1) % self.veh_count
    #             attempts += 1
            
    #         # If we checked all vehicles and they're all done, use the first one
    #         # (this should only happen if all vehicles are done)
    #         if attempts == self.veh_count:
    #             v_idx = 0
                
    #         # Assign selected vehicle for this batch
    #         self.cur_veh_idx[b, 0] = v_idx
            
    #         # Update next_vehicle_idx for next call (move to next vehicle)
    #         self.next_vehicle_idx[b] = (v_idx + 1) % self.veh_count
        
    #     # Get vehicle state based on selected indices
    #     self.cur_veh = self.vehicles.gather(1,
    #         self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE))
        
    #     # Update vehicle mask
    #     self.cur_veh_mask = self.mask.gather(1,
    #         self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))
        
    #     # Log vehicle selection
    #     for b in range(batch_size):
    #         self.logger.info(f"VEH_SELECT | B{b:03d} | Selected V{self.cur_veh_idx[b, 0].item()} | " +
    #                         f"Time={self.cur_veh[b, 0, 3].item():.1f} | " +
    #                         f"Cap={self.cur_veh[b, 0, 2].item():.1f}") 



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
        self.cust_mask = self.init_cust_mask
        self.new_customers = True
        self.served = self.nodes.new_zeros((self.minibatch_size, self.nodes_count), dtype=torch.bool)
        
        # Initialize late nodes tracking
        self.late_nodes = self.nodes.new_zeros((self.minibatch_size, self.nodes_count), dtype=torch.bool)
        self.node_lateness_count = 0
        
        # Initialize vehicle visit tracking for capacity debugging
        self.vehicle_visit_count = torch.zeros((self.minibatch_size, self.veh_count), 
                                            dtype=torch.long, device=self.nodes.device)

        self.mask = self.nodes.new_zeros((self.minibatch_size, self.veh_count, self.nodes_count), dtype=torch.bool)
        if self.cust_mask is not None:
            self.mask = self.cust_mask[:, None, :].repeat(1, self.veh_count, 1)
       
        # Initialize urgent deadlines for vehicles as infinity
        # self.urgent_deadlines = torch.full((self.minibatch_size, self.veh_count), float('inf'), 
        #                                  device=self.nodes.device)

        self.next_vehicle_idx = torch.zeros((self.minibatch_size,), dtype=torch.long, device=self.nodes.device)
       
        self.cur_veh_idx = self.nodes.new_zeros((self.minibatch_size, 1), dtype=torch.int64)
        self.cur_veh = self.vehicles.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE))

        self.cur_veh_mask = self.mask.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))
        
        self.vehicle_routes = self.nodes.new_zeros(
            (self.minibatch_size, self.nodes_count),
            dtype=torch.long
        ) - 1
        
        # Reset batch tracking
        self.batch_costs = torch.zeros(self.minibatch_size)
        self.batch_served = torch.zeros(self.minibatch_size, dtype=torch.long)
        
        
    def step(self, cust_idx):
        # 1) ظرفیت‌‌سنجی
        for b in range(self.minibatch_size):
            node = cust_idx[b,0].item()
            if node > 0 and self.cur_veh[b,0,2] < 1:
                self.logger.info(f"CAPACITY ERROR | B{b:03d} | node {node} بدون ظرفیت")
                cust_idx[b,0] = 0

        # 2) به‌روزرسانی وضعیت
        dest = self.nodes.gather(1, cust_idx[:,:,None].expand(-1,-1,self.CUST_FEAT_SIZE))
        dist, node_late, node_on_time = self._update_vehicles(dest, cust_idx)
        self._update_done(cust_idx)
        self._update_mask(cust_idx)
        self._update_cur_veh()

        # 3) پاداش لحظه‌ای
        reward = (-self.spoilage_penalty * node_late
                - self.dist_penalty_coef * dist)

        # 4) پایان اپیزود
        if self.done:
            feasible_mask = ~self.infeasible_nodes
            feasible_mask[:,0] = False               # دپو را حذف کن
            
            # A) late در سطح گره
            pickup_late_cnt  = (self.late_nodes & feasible_mask).float().sum(dim=1, keepdim=True)

            

            late_at_depot_cnt  = torch.zeros_like(pickup_late_cnt )            # [B,1]
            late_at_depot_bool = torch.zeros_like(self.late_nodes, dtype=torch.bool)

            arrival_in_depot = self.vehicles[:, :, 3]       # [B,V]  (اسکالرِ زمانِ برگشت)

            for b in range(self.minibatch_size):
                for v in range(self.veh_count):

                    # گره‌هایی که وسیلهٔ v در این batch حمل کرده (به جز دپو)
                    nodes_v = torch.nonzero(
                        (self.vehicle_routes[b] == v) &
                        (torch.arange(self.nodes_count, device=self.nodes.device) > 0),
                        as_tuple=False
                    ).squeeze(-1)

                    if nodes_v.numel() == 0:
                        continue

                    t_dep  = arrival_in_depot[b, v].item()          # ← float
                    spoil  = self.nodes[b, nodes_v, 3]              # [K] spoilage times

                    late_flags = (t_dep > spoil)                    # [K] bool

                    late_at_depot_bool[b, nodes_v] |= late_flags

                    # جمع تعداد
                    late_at_depot_cnt[b, 0] += late_flags.sum().float()


            # C) unserved
            mask_nodes = torch.arange(self.nodes_count, device=self.nodes.device) > 0  # [N]  True برای نودهای غیر-دپو
            feasible_non_depot = feasible_mask & mask_nodes           # [B,N]

            unserved_cnt = (~self.served & feasible_non_depot) \
                            .float().sum(dim=1, keepdim=True)

            # D) idle vehicles
            idle_cnt = (self.vehicle_visit_count==0).sum(dim=1, keepdim=True)  # [B,1]
            idle_penalty = -self.idle_penalty_coef * idle_cnt

            
            # J_node   = -self.additional_late_penalty * node_late_cnt
            J_depot  = -self.additional_late_penalty * late_at_depot_cnt
            J_unserv = -self.unserved_penalty       * unserved_cnt
            # J_dist   = -self.dist_penalty_coef      * total_dist
            J_idle   = idle_penalty                 # [B,1]

            final_reward = ( J_depot + J_unserv + J_idle )
            reward += final_reward 

            # موفقیت کامل
            # cust_mask      = (torch.arange(self.nodes_count, device=self.nodes.device) > 0)
            # feasible_cust  = feasible_mask & cust_mask          # فقط  قابل‌خدمت
            # good_nodes     = self.served & (~self.late_nodes) & (~late_at_depot_bool)

                
            # total_feasible = feasible_cust.sum(dim=1, keepdim=True).float()  # [B,1]
            # total_success = (feasible_cust & good_nodes).sum(dim=1, keepdim=True).float()  # [B,1]
            
            # # جلوگیری از تقسیم بر صفر
            # success_ratio = torch.where(total_feasible > 0, 
            #                           total_success / total_feasible, 
            #                           torch.ones_like(total_feasible))  # [B,1]

            # # Success bonus برای threshold 80%
            # success_bonus_80 = self.success_bonus * (success_ratio >= 0.8).float()
            # reward += success_bonus_80
            
            if self.logger:  # فقط وجود logger را چک کن
                for b in range(self.minibatch_size):
                    # محاسبه آمار تکمیلی
                    total_feasible = feasible_non_depot[b].sum().item()
                    total_served = self.served[b, 1:].sum().item()  # حذف depot از شمارش
                    serving_rate = (total_served / total_feasible * 100) if total_feasible > 0 else 0
                    
                    # لاگ کامل reward breakdown
                    self.logger.info(
                        f"[EPISODE_END] B{b:03d} | "
                        f"Served={total_served}/{total_feasible} ({serving_rate:.1f}%) | "
                        f"PickupLate={pickup_late_cnt[b,0]:.0f} | "
                        f"DepotLate={late_at_depot_cnt[b,0]:.0f} | "
                        f"Unserved={unserved_cnt[b,0]:.0f} | "
                        f"Idle={idle_cnt[b,0]:.0f} | "
                        f"Rewards: Depot={J_depot[b,0]:.1f} Unserv={J_unserv[b,0]:.1f} Idle={J_idle[b,0]:.1f} | "
                        f"FinalReward={reward[b,0]:.1f}"
                    )
            self.last_reward = reward.clone()

        if reward.dim()==1:
            reward = reward.unsqueeze(-1)
        return reward


    def get_state(self):
        """Return current state"""
        return None

    def state_dict(self, dest_dict=None):
        """Save state to dictionary"""
        if dest_dict is None:
            dest_dict = {
                "vehicles": self.vehicles,
                "veh_done": self.veh_done,
                "served": self.served,
                "mask": self.mask,
                "cur_veh_idx": self.cur_veh_idx,
                "vehicle_routes": self.vehicle_routes,
                "late_nodes": self.late_nodes
            }
        else:
            dest_dict["vehicles"].copy_(self.vehicles)
            dest_dict["veh_done"].copy_(self.veh_done)
            dest_dict["served"].copy_(self.served)
            dest_dict["mask"].copy_(self.mask)
            dest_dict["cur_veh_idx"].copy_(self.cur_veh_idx)
            dest_dict["vehicle_routes"].copy_(self.vehicle_routes)
            dest_dict["late_nodes"].copy_(self.late_nodes)
        return dest_dict

    def load_state_dict(self, state_dict):
        """Load state from dictionary"""
        self.vehicles.copy_(state_dict["vehicles"])
        self.veh_done.copy_(state_dict["veh_done"])
        self.served.copy_(state_dict["served"])
        self.mask.copy_(state_dict["mask"])
        self.cur_veh_idx.copy_(state_dict["cur_veh_idx"])
        self.vehicle_routes.copy_(state_dict["vehicle_routes"])
        
        if "late_nodes" in state_dict:
            self.late_nodes.copy_(state_dict["late_nodes"])
            
        self.cur_veh = self.vehicles.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.VEH_STATE_SIZE))
        self.cur_veh_mask = self.mask.gather(1,
            self.cur_veh_idx[:, :, None].expand(-1, -1, self.nodes_count))

    def call_ortools(data):
        routes = ort_solve(data)
        # حالا هزینه‌ها و پنالتی‌ها را حساب می‌کنیم

        costs = []
        penalties = []
        for b_idx, nodes in enumerate(data.nodes_gen()):
            route_set = routes[b_idx]
            if not route_set:
                costs.append(1e8)
                penalties.append(0)
                continue

            total_cost = 0
            penalty = 0
            depot = nodes[0, :2]
            veh_speed = data.veh_speed

            for route in route_set:
                if not route:
                    # ماشین بیکار → جریمه Idle
                    penalty += 10 * 20  # α₅ * scale = 200
                    continue

                pos = depot
                time = 0
                for n in route:
                    next_pos = nodes[n, :2]
                    dist = (next_pos - pos).pow(2).sum().sqrt()
                    time += dist / veh_speed
                    total_cost += 0.05 * dist.item()

                    # بررسی زنده‌ماندن بیمار (late penalty)
                    survival = nodes[n, 3].item()
                    delivery_time = time + (depot - next_pos).pow(2).sum().sqrt() / veh_speed
                    if delivery_time > survival:
                        penalty += (delivery_time - survival) * 1.0  # α₃
                    pos = next_pos

            costs.append(total_cost)
            penalties.append(penalty)

        return routes, costs, penalties
