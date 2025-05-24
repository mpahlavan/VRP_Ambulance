from marpdan import AttentionLearner
from marpdan.problems import PVRP_Dataset, PVRP_Environment
from marpdan.externals import ort_solve
from marpdan.utils import *
from marpdan.dep import matplotlib as mpl, pyplot as plt
import numpy as np
import torch
import time, os

SEED = 12348877555
BATCH_SIZE = 400
mpl.rcParams["axes.titlesize"] = 20

class PVRPAnalyzer:
    def __init__(self, args):
        self.problem_type = args.problem_type
        self.n_customers = args.customers_count
        self.n_vehicles = args.vehicles_count
        self.veh_capa = args.veh_capa
        self.veh_speed = args.veh_speed
        self.min_cust_count = args.min_cust_count
        self.cust_loc_range = args.loc_range
        self.horizon = args.horizon
        self.spoilage_range = args.spoilage_range
        date = "250524-1621"
        self.MODEL_PATH = f"./output/PVRPn{args.customers_count}m{args.vehicles_count}_{date}/chkpt_ep{args.epoch_count}.pyth"
        self.learner = self._load_model()
        
    def _load_model(self):
        try:
            chkpt = torch.load(self.MODEL_PATH, map_location="cpu")
            learner = AttentionLearner(
                cust_feat_size=PVRP_Dataset.CUST_FEAT_SIZE,
                veh_state_size=PVRP_Environment.VEH_STATE_SIZE
            )
            learner.load_state_dict(chkpt["model"])
            learner.eval()
            return learner
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {self.MODEL_PATH}")
    
    def generate_data(self):
        torch.manual_seed(SEED)
        data = PVRP_Dataset.generate(
            BATCH_SIZE, 
            self.n_customers, 
            self.n_vehicles, 
            self.veh_capa,
            self.veh_speed,
            self.min_cust_count,
            self.cust_loc_range,
            self.horizon,
            self.spoilage_range
        )
        ref_routes = ort_solve(data)
        data.normalize()
        return data, ref_routes

    # MODIFIED: تابع محاسبه هزینه‌های مرجع با در نظر گرفتن پنالتی‌های پایان اپیزود
    def calculate_reference_costs(self, data, ref_routes):
        """محاسبه هزینه‌های مسیرهای مرجع با در نظر گرفتن پنالتی‌های پایان اپیزود"""
        ref_costs = []
        
        # MODIFIED: متغیرهای آماری برای ردیابی عملکرد مرجع
        ref_unserved_total = 0
        ref_late_total = 0
        ref_served_total = 0
        
        for batch_idx, routes in enumerate(ref_routes):
            single_env = PVRP_Environment(
                data,
                nodes=data.nodes[batch_idx:batch_idx+1],
                cust_mask=data.cust_mask[batch_idx:batch_idx+1] if data.cust_mask is not None else None
            )
            
            # MODIFIED: اضافه کردن ویژگی last_reward
            if not hasattr(single_env, 'last_reward'):
                single_env.last_reward = torch.zeros((single_env.minibatch_size, 1), device=single_env.nodes.device)
            
            single_env.reset()
            rewards = []
            
            for route in routes:
                for node in route:
                    node_tensor = torch.tensor([[node]], device=data.nodes.device, dtype=torch.long)
                    reward = single_env.step(node_tensor)
                    rewards.append(reward)
            
            # MODIFIED: بررسی اتمام اپیزود و محاسبه پنالتی‌های نهایی
            if not single_env.done:
                depot_action = torch.tensor([[0]], device=data.nodes.device, dtype=torch.long)
                final_reward = single_env.step(depot_action)
                rewards.append(final_reward)
                single_env.last_reward = final_reward.clone()
            
            # جمع‌آوری آمار
            for b in range(single_env.minibatch_size):
                ref_served_total += single_env.served[b].sum().item()
                if hasattr(single_env, 'late_nodes'):
                    ref_late_total += single_env.late_nodes[b].sum().item()
                
                if single_env.init_cust_mask is not None:
                    effective_mask = ~single_env.init_cust_mask[b].bool()
                    unserved = (effective_mask & ~single_env.served[b] & ~single_env.infeasible_nodes[b]).sum().item()
                else:
                    unserved = (~single_env.served[b] & ~single_env.infeasible_nodes[b]).sum().item()
                ref_unserved_total += unserved
            
            # محاسبه هزینه کل
            if rewards:
                ref_costs.append(-torch.stack(rewards).sum())
            else:
                ref_costs.append(torch.tensor(float('inf'), device=data.nodes.device))
        
        # MODIFIED: چاپ آمار مرجع
        print(f"\nORTools Reference Stats:")
        print(f"Served Nodes: {ref_served_total}, Late Nodes: {ref_late_total}, Unserved Nodes: {ref_unserved_total}")
        print(f"Average per instance - Served: {ref_served_total/len(ref_routes):.2f}, Late: {ref_late_total/len(ref_routes):.2f}, Unserved: {ref_unserved_total/len(ref_routes):.2f}")
        
        return torch.stack(ref_costs)

    @staticmethod
    def plot_pvrp_instance(ax, nodes, routes, title):
        ax.plot(nodes[0,0].item(), nodes[0,1].item(), 'ks', markersize=10, label='Depot')
        scatter = ax.scatter(nodes[1:,0], nodes[1:,1], 
                             c=nodes[1:,3], 
                             cmap='RdYlGn',
                             label='Customers (color=spoilage time)')
        plt.colorbar(scatter, ax=ax)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
        for route, color in zip(routes, colors):
            route = [0] + route + [0]
            points = nodes[route]
            ax.plot(points[:,0], points[:,1], '-', 
                   color=color, alpha=0.7, 
                   label=f'Route (len={len(route)-2})')
        ax.set_title(title)
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # MODIFIED: تابع جدید برای نمایش اطلاعات تفصیلی هزینه‌ها
    def display_cost_breakdown(self, env, is_reference=False):
        """نمایش جزئیات هزینه‌ها برای یک محیط"""
        
        # شناسایی اجزای هزینه‌ها (تقریبی)
        total_distance = 0
        for v in range(env.veh_count):
            v_indices = []
            for b in range(env.minibatch_size):
                # دریافت گره‌های بازدید شده توسط این خودرو
                nodes = (env.vehicle_routes[b] == v).nonzero(as_tuple=True)[0].tolist()
                if nodes:
                    # اضافه کردن دپو به عنوان اولین و آخرین توقف
                    route = [0] + nodes + [0]
                    # محاسبه فاصله مسیر
                    positions = env.nodes[b, route, :2]
                    segments = positions[1:] - positions[:-1]
                    route_dist = segments.norm(dim=1).sum()
                    total_distance += route_dist
        
        # محاسبه تقریبی هزینه‌های مختلف
        distance_penalty = -env.dist_penalty_coef * total_distance
        
        # گره‌های خدمت داده نشده
        unserved_count = 0
        for b in range(env.minibatch_size):
            if env.init_cust_mask is not None:
                effective_mask = ~env.init_cust_mask[b].bool()
                unserved = (effective_mask & ~env.served[b] & ~env.infeasible_nodes[b]).sum().item()
            else:
                unserved = (~env.served[b] & ~env.infeasible_nodes[b]).sum().item()
            unserved_count += unserved
        unserved_penalty = -env.unserved_penalty * unserved_count
        
        # گره‌های با دیرکرد
        late_count = 0
        if hasattr(env, 'late_nodes'):
            for b in range(env.minibatch_size):
                late_count += env.late_nodes[b].sum().item()
        late_penalty = -env.spoilage_penalty * late_count - env.additional_late_penalty * late_count
        
        # پاداش جمع‌آوری به‌موقع
        served_count = 0
        for b in range(env.minibatch_size):
            served_count += env.served[b].sum().item()
        ontime_count = served_count - late_count
        pickup_bonus = env.pickup_bonus_coef * ontime_count
        
        # خودروهای بیکار
        idle_vehicles = 0
        for b in range(env.minibatch_size):
            for v in range(env.veh_count):
                if not (env.vehicle_routes[b] == v).any():
                    idle_vehicles += 1
        idle_penalty = -env.idle_penalty_coef * idle_vehicles
        
        # ظرفیت استفاده نشده
        unused_capacity = 0
        for b in range(env.minibatch_size):
            for v in range(env.veh_count):
                v_route = (env.vehicle_routes[b] == v).sum().item()
                if v_route > 0:  # فقط خودروهای استفاده شده
                    unused_capacity += (env.veh_capa - v_route) / env.veh_capa
        capacity_penalty = -env.capacity_usage_coef * unused_capacity
        
        # مجموع کل
        total_cost = (distance_penalty + unserved_penalty + late_penalty + 
                     pickup_bonus + idle_penalty + capacity_penalty)
        
        # چاپ نتایج
        model_type = "ORTools" if is_reference else "Learned"
        print(f"\n{model_type} Cost Breakdown:")
        print(f"{'Component':<20} {'Value':<10} {'Description'}")
        print("-" * 60)
        print(f"{'Distance Penalty':<20} {distance_penalty:10.2f} (Total distance: {total_distance:.2f})")
        print(f"{'Unserved Penalty':<20} {unserved_penalty:10.2f} (Unserved: {unserved_count})")
        print(f"{'Late Penalty':<20} {late_penalty:10.2f} (Late: {late_count})")
        print(f"{'Pickup Bonus':<20} {pickup_bonus:10.2f} (On-time: {ontime_count})")
        print(f"{'Idle Penalty':<20} {idle_penalty:10.2f} (Idle vehicles: {idle_vehicles})")
        print(f"{'Capacity Penalty':<20} {capacity_penalty:10.2f} (Unused capacity ratio: {unused_capacity:.2f})")
        print(f"{'TOTAL COST':<20} {total_cost:10.2f}")
        
        return {
            'distance': distance_penalty,
            'unserved': unserved_penalty,
            'late': late_penalty,
            'pickup': pickup_bonus,
            'idle': idle_penalty,
            'capacity': capacity_penalty,
            'total': total_cost,
            'served_count': served_count,
            'late_count': late_count,
            'unserved_count': unserved_count
        }

    # MODIFIED: تابع جدید برای تحلیل و مقایسه هزینه‌ها
    def analyze_and_visualize(self):
        """تحلیل و بصری‌سازی کامل راه‌حل‌های PVRP با جزئیات هزینه‌ها"""
        
        # تولید داده‌ها و دریافت راه‌حل‌های مرجع
        data, ref_routes = self.generate_data()
        ref_costs = self.calculate_reference_costs(data, ref_routes)
        
        # MODIFIED: ایجاد متغیرهای آماری برای مدل یادگیری
        learned_costs = []
        learned_unserved_total = 0
        learned_late_total = 0
        learned_served_total = 0
        
        print("\nAnalyzing learned model...")
        for batch_idx in range(data.nodes.size(0)):
            # ایجاد محیط برای این نمونه
            single_data = PVRP_Dataset(
                data.veh_count,
                data.veh_capa,
                data.veh_speed,
                data.nodes[batch_idx:batch_idx+1].clone(),
                None if data.cust_mask is None else data.cust_mask[batch_idx:batch_idx+1].clone()
            )
            
            single_env = PVRP_Environment(single_data)
            
            # MODIFIED: اضافه کردن ویژگی last_reward اگر وجود ندارد
            if not hasattr(single_env, 'last_reward'):
                single_env.last_reward = torch.zeros((single_env.minibatch_size, 1), device=single_env.nodes.device)
            
            single_env.reset()
            
            # اجرای مدل یادگیری
            with torch.no_grad():
                learned_actions, _, learned_rewards = self.learner(single_env)
            
            # MODIFIED: بررسی اتمام اپیزود و محاسبه پنالتی‌های نهایی
            if not single_env.done:
                # اجرای یک مرحله برای به پایان رساندن اپیزود و محاسبه پنالتی‌های پایان
                depot_action = torch.tensor([[0]], device=data.nodes.device, dtype=torch.long)
                final_reward = single_env.step(depot_action)
                learned_rewards.append(final_reward)
                single_env.last_reward = final_reward.clone()
            
            # محاسبه هزینه کل
            full_reward = torch.stack(learned_rewards).sum()
            learned_costs.append(-full_reward.item())
            
            # جمع‌آوری آمار
            for b in range(single_env.minibatch_size):
                learned_served_total += single_env.served[b].sum().item()
                if hasattr(single_env, 'late_nodes'):
                    learned_late_total += single_env.late_nodes[b].sum().item()
                
                if single_env.init_cust_mask is not None:
                    effective_mask = ~single_env.init_cust_mask[b].bool()
                    unserved = (effective_mask & ~single_env.served[b] & ~single_env.infeasible_nodes[b]).sum().item()
                else:
                    unserved = (~single_env.served[b] & ~single_env.infeasible_nodes[b]).sum().item()
                learned_unserved_total += unserved
        
        # MODIFIED: چاپ آمار و مقایسه
        print(f"\nLearned Model Stats:")
        print(f"Served Nodes: {learned_served_total}, Late Nodes: {learned_late_total}, Unserved Nodes: {learned_unserved_total}")
        print(f"Average per instance - Served: {learned_served_total/len(learned_costs):.2f}, Late: {learned_late_total/len(learned_costs):.2f}, Unserved: {learned_unserved_total/len(learned_costs):.2f}")
        
        # تبدیل به تنسور و محاسبه شکاف‌ها
        learned_costs = torch.tensor(learned_costs, device=ref_costs.device)
        gaps = learned_costs / ref_costs - 1
        
        # مقایسه هزینه‌ها
        print("\nCost comparison:")
        print(f"Average ORTools Cost: {ref_costs.mean().item():.2f}")
        print(f"Average Learned Cost: {learned_costs.mean().item():.2f}")
        print(f"Mean gap: {gaps.mean():.2%}, Max gap: {gaps.max():.2%}, Min gap: {gaps.min():.2%}")
        
        # مرتب‌سازی بر اساس شکاف
        gaps, sub_idx = gaps.sort()
        
        # انتخاب نمونه‌ها برای نمایش
        display_indices = torch.cat((
            sub_idx[:4],                            # بهترین موارد
            sub_idx[BATCH_SIZE//2-2:BATCH_SIZE//2+2], # موارد میانه
            sub_idx[-4:]                           # بدترین موارد
        ))
        
        # اضافه کردن پوشه خروجی
        output_dir = f"results/pvrp_n{self.n_customers}m{self.n_vehicles}_{time.strftime('%y%m%d-%H%M')}"
        os.makedirs(output_dir, exist_ok=True)
        
        # MODIFIED: تحلیل دقیق مقایسه‌ای برای یک نمونه
        # انتخاب یک نمونه از بهترین، میانه و بدترین موارد برای تحلیل دقیق
        analysis_indices = [sub_idx[0], sub_idx[BATCH_SIZE//2], sub_idx[-1]]
        
        for analysis_idx in analysis_indices:
            idx = analysis_idx.item()
            print(f"\n{'='*60}")
            print(f"Detailed Analysis for Sample {idx} (Gap: {gaps[torch.where(sub_idx == idx)[0][0]]:.2%})")
            print(f"{'='*60}")
            
            # تنظیم محیط ORTools برای تحلیل
            ref_env = PVRP_Environment(
                data,
                nodes=data.nodes[idx:idx+1],
                cust_mask=data.cust_mask[idx:idx+1] if data.cust_mask is not None else None
            )
            if not hasattr(ref_env, 'last_reward'):
                ref_env.last_reward = torch.zeros((ref_env.minibatch_size, 1), device=ref_env.nodes.device)
            
            ref_env.reset()
            for route in ref_routes[idx]:
                for node in route:
                    node_tensor = torch.tensor([[node]], device=data.nodes.device, dtype=torch.long)
                    _ = ref_env.step(node_tensor)
            
            if not ref_env.done:
                _ = ref_env.step(torch.tensor([[0]], device=data.nodes.device, dtype=torch.long))
            
            # تنظیم محیط مدل یادگیری برای تحلیل
            learned_env = PVRP_Environment(
                PVRP_Dataset(
                    data.veh_count,
                    data.veh_capa,
                    data.veh_speed,
                    data.nodes[idx:idx+1].clone(),
                    None if data.cust_mask is None else data.cust_mask[idx:idx+1].clone()
                )
            )
            if not hasattr(learned_env, 'last_reward'):
                learned_env.last_reward = torch.zeros((learned_env.minibatch_size, 1), device=learned_env.nodes.device)
            
            learned_env.reset()
            with torch.no_grad():
                learned_actions, _, _ = self.learner(learned_env)
            
            if not learned_env.done:
                _ = learned_env.step(torch.tensor([[0]], device=data.nodes.device, dtype=torch.long))
            
            # تحلیل هزینه‌ها
            ref_costs_detail = self.display_cost_breakdown(ref_env, is_reference=True)
            learned_costs_detail = self.display_cost_breakdown(learned_env, is_reference=False)
            
            # مقایسه هزینه‌ها
            print(f"\nComponent Comparison:")
            print(f"{'Component':<15} {'ORTools':<10} {'Learned':<10} {'Diff':<10} {'% Diff':<10}")
            print("-" * 60)
            
            for key in ref_costs_detail:
                if key in learned_costs_detail and isinstance(ref_costs_detail[key], (int, float)):
                    diff = learned_costs_detail[key] - ref_costs_detail[key]
                    if ref_costs_detail[key] != 0:
                        pct_diff = diff / abs(ref_costs_detail[key]) * 100
                    else:
                        pct_diff = float('inf') if diff != 0 else 0
                    
                    print(f"{key:<15} {ref_costs_detail[key]:10.2f} {learned_costs_detail[key]:10.2f} {diff:10.2f} {pct_diff:10.1f}%")
        
        # بصری‌سازی نمونه‌های انتخاب شده
        for i, idx in enumerate(display_indices):
            cust = data.nodes[idx]
            ref_route = ref_routes[idx]
            ref_cost = ref_costs[idx]
            
            # تنظیم محیط برای مدل یادگیری
            single_data = PVRP_Dataset(
                data.veh_count,
                data.veh_capa,
                data.veh_speed,
                data.nodes[idx:idx+1].clone(),
                None if data.cust_mask is None else data.cust_mask[idx:idx+1].clone()
            )
            
            single_env = PVRP_Environment(single_data)
            if not hasattr(single_env, 'last_reward'):
                single_env.last_reward = torch.zeros((single_env.minibatch_size, 1), device=single_env.nodes.device)
            
            single_env.reset()
            
            # اجرای مدل یادگیری
            with torch.no_grad():
                learned_actions, _, learned_rewards = self.learner(single_env)
            
            # بررسی اتمام اپیزود و محاسبه پنالتی‌های نهایی
            if not single_env.done:
                depot_action = torch.tensor([[0]], device=data.nodes.device, dtype=torch.long)
                final_reward = single_env.step(depot_action)
                learned_rewards.append(final_reward)
            
            # محاسبه هزینه کل
            model_cost = -torch.stack(learned_rewards).sum().item()
            gap = model_cost / ref_cost - 1
            
            # استخراج مسیرها
            learned_routes = [[] for _ in range(single_env.veh_count)]
            for action in learned_actions:
                veh_idx, node_idx = action
                v = veh_idx[0].item()
                n = node_idx[0].item()
                if n > 0:  # به جز دپو
                    learned_routes[v].append(n)
            
            # حذف مسیرهای خالی
            learned_routes = [route for route in learned_routes if route]
            
            # شمارش گره‌های خدمت داده شده، دیرکرد و خدمت داده نشده
            ref_served = set([node for route in ref_route for node in route])
            learned_served = set([node for route in learned_routes for node in route])
            all_nodes = set(range(1, len(cust)))  # به جز دپو (0)
            
            ref_unserved = all_nodes - ref_served
            learned_unserved = all_nodes - learned_served
            
            # ایجاد بصری‌سازی با اطلاعات اضافی
            fig, (ref_ax, ax) = plt.subplots(1, 2, figsize=(20, 8))
            
            # اضافه کردن آمار خاص نمونه به عناوین
            ref_title = (f"ORTools (cost = {ref_cost:.1f})\n"
                         f"Served: {len(ref_served)}, Unserved: {len(ref_unserved)}")
            
            learned_title = (f"Learned (cost = {model_cost:.1f}, gap = {gap:.0%})\n"
                            f"Served: {len(learned_served)}, Unserved: {len(learned_unserved)}")
            
            # رسم با اطلاعات پیشرفته
            self.plot_pvrp_instance(ref_ax, cust, ref_route, ref_title)
            self.plot_pvrp_instance(ax, cust, learned_routes, learned_title)
            
            # افزودن گره‌های خدمت داده نشده به عنوان علامت‌های 'X'
            for node in ref_unserved:
                ref_ax.plot(cust[node, 0].item(), cust[node, 1].item(), 'rx', markersize=10)
            
            for node in learned_unserved:
                ax.plot(cust[node, 0].item(), cust[node, 1].item(), 'rx', markersize=10)
            
            # افزودن زمان‌های فساد به عنوان یادداشت‌های متنی
            for node in range(1, len(cust)):
                spoilage = cust[node, 3].item()
                # قالب‌بندی به عنوان یادداشت برای هر دو نمودار
                ref_ax.annotate(f"{spoilage:.2f}", (cust[node, 0].item(), cust[node, 1].item()), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                ax.annotate(f"{spoilage:.2f}", (cust[node, 0].item(), cust[node, 1].item()), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            fig.tight_layout()
            file_path = f"{output_dir}/pvrp_routes_n{self.n_customers}m{self.n_vehicles}_{i:02}_{100*gap:.0f}.pdf"
            fig.savefig(file_path, bbox_inches='tight')
        
        plt.show()

def main(args):
    analyzer = PVRPAnalyzer(args)
    analyzer.analyze_and_visualize()

if __name__ == "__main__":
    main(parse_args())