import numpy as np
import matplotlib.pyplot as plt
import random
import copy

# ==========================================
# 1. Configuration
# ==========================================
L = 40              
N_PARTICLES = 400   
N_COLORS = 4        
STEPS = 500         

# Thermodynamic Costs (The Physics)
Q_FREE = 0.1        # 自由移动
Q_SYNERGY = 0.01    # 协同 (同色)
Q_SCATTER = 1.0     # 普通散射 (异色 - 无关)
Q_FRUSTRATION = 10.0 # 阻挫/排斥 (异色 - 冲突) -> 比如 红色 vs 蓝色

# Conflict Definition
# 红色(0) 和 蓝色(1) 是死敌
CONFLICT_PAIRS = [(0, 1), (1, 0)] 

COLOR_MAP = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}

# ==========================================
# 2. Core Logic
# ==========================================

class Particle:
    def __init__(self, p_id, color, x, y):
        self.id = p_id
        self.color = color
        self.x = x
        self.y = y

class SemanticGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.full((width, height), -1, dtype=int)
        self.particles = []
        
    def add_particle(self, p):
        self.particles.append(p)
        self.grid[p.x, p.y] = p.id

    def get_particle_at(self, x, y):
        p_id = self.grid[x, y]
        if p_id == -1: return None
        for p in self.particles:
            if p.id == p_id: return p
        return None

    def move_particle(self, p, nx, ny):
        self.grid[p.x, p.y] = -1
        p.x, p.y = nx, ny
        self.grid[nx, ny] = p.id
        
    def swap_particles(self, p1, p2):
        self.grid[p1.x, p1.y] = p2.id
        self.grid[p2.x, p2.y] = p1.id
        p1.x, p1.y, p2.x, p2.y = p2.x, p2.y, p1.x, p1.y

# ==========================================
# 3. Physics Engine with Frustration
# ==========================================

def run_step_frustration(sim):
    step_heat = 0.0
    
    particles = list(sim.particles)
    random.shuffle(particles)
    
    for p in particles:
        # Try move
        dx, dy = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        tx, ty = p.x + dx, p.y + dy
        
        # Wall check
        if not (0 <= tx < L and 0 <= ty < L):
            step_heat += Q_SCATTER 
            continue
            
        target = sim.get_particle_at(tx, ty)
        
        if target is None:
            # Free move
            sim.move_particle(p, tx, ty)
            step_heat += Q_FREE
            
        elif target.color == p.color:
            # Synergy
            sim.swap_particles(p, target)
            step_heat += Q_SYNERGY
            
        else:
            # Heterogeneous Collision
            # Check for Frustration (Conflict)
            if (p.color, target.color) in CONFLICT_PAIRS:
                # 剧烈排斥/阻挫
                # Move fails + High Heat
                step_heat += Q_FRUSTRATION
            else:
                # 普通散射
                step_heat += Q_SCATTER
                
    return step_heat

# ==========================================
# 4. Scenarios
# ==========================================

def generate_mixed_conflict(n_particles):
    """
    场景 C: 强制混合冲突粒子 (红色和蓝色)
    模拟脏数据/逻辑矛盾
    """
    sim = SemanticGrid(L, L)
    # 严格直方图: 各 100 个
    particles_per_color = n_particles // N_COLORS
    colors = []
    for c in range(N_COLORS): colors.extend([c] * particles_per_color)
    
    # 策略：特意把红色和蓝色撒在同一个区域，制造麻烦
    positions = []
    occupied = set()
    
    # 区域划分
    # 区域 1 (Conflict Zone): 红色 + 蓝色
    # 区域 2 (Safe Zone): 绿色 + 橙色
    
    center_conflict = (L//2, L//2)
    center_safe_1 = (L//4, L//4) # 只是为了填满
    center_safe_2 = (3*L//4, 3*L//4)
    
    all_coords = [(x,y) for x in range(L) for y in range(L)]
    random.shuffle(all_coords) # 用于非冲突粒子的随机填补
    
    pos_idx = 0
    
    # 分配位置
    color_positions = {c: [] for c in range(N_COLORS)}
    
    for c in range(N_COLORS):
        count = 0
        while count < particles_per_color:
            if c in [0, 1]: # Red & Blue (Enemies)
                # 强制它们挤在中间，增加接触率
                rx = int(random.gauss(center_conflict[0], 5.0)) # 比较密
                ry = int(random.gauss(center_conflict[1], 5.0))
                rx, ry = max(0, min(L-1, rx)), max(0, min(L-1, ry))
                pos = (rx, ry)
                if pos not in occupied:
                    occupied.add(pos)
                    sim.add_particle(Particle(len(sim.particles), c, rx, ry))
                    count += 1
            else:
                # Green & Orange (Bystanders) - 扔远点或者随机
                # 简单起见，随机填空位，或者聚在角落不干扰
                while True:
                    cand = all_coords[pos_idx]
                    pos_idx += 1
                    if cand not in occupied:
                        occupied.add(cand)
                        sim.add_particle(Particle(len(sim.particles), c, cand[0], cand[1]))
                        count += 1
                        break
    return sim

def generate_random(n_particles):
    # 复用之前的随机生成
    sim = SemanticGrid(L, L)
    particles_per_color = n_particles // N_COLORS
    colors = []
    for c in range(N_COLORS): colors.extend([c] * particles_per_color)
    random.shuffle(colors)
    
    all_coords = [(x, y) for x in range(L) for y in range(L)]
    positions = random.sample(all_coords, n_particles)
    
    for i in range(n_particles):
        sim.add_particle(Particle(i, colors[i], positions[i][0], positions[i][1]))
    return sim

# ==========================================
# 5. Main
# ==========================================

def run_frustration_experiment():
    print("Running Experiment 3B: Semantic Frustration...")
    
    # 1. Run Random (Baseline Noise)
    sim_rnd = generate_random(N_PARTICLES)
    q_rnd_history = []
    q_curr = 0
    for _ in range(STEPS):
        h = run_step_frustration(sim_rnd)
        q_curr += h
        q_rnd_history.append(q_curr)
        
    # 2. Run Conflict (Frustration)
    sim_conf = generate_mixed_conflict(N_PARTICLES)
    q_conf_history = []
    q_curr = 0
    
    # Snapshot for visualization (Initial)
    sim_conf_initial = copy.deepcopy(sim_conf) # Just coordinate copy actually needed
    
    for _ in range(STEPS):
        h = run_step_frustration(sim_conf)
        q_curr += h
        q_conf_history.append(q_curr)
        
    # =================
    # Visualization
    # =================
    plt.rcParams.update({'font.size': 10})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Heat Comparison
    ax1.plot(q_conf_history, 'purple', lw=2, label='Scenario C: Conflict (Frustration)')
    ax1.plot(q_rnd_history, 'r--', lw=2, label='Scenario A: Noise (Baseline)')
    ax1.fill_between(range(STEPS), q_rnd_history, q_conf_history, color='purple', alpha=0.1)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Cumulative Heat (Q)')
    ax1.set_title('Thermodynamic Penalty of Frustration\n(Conflict Generates Excess Heat)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: The Conflict Zone (Visualizing Domain Walls)
    # Plot final state of Conflict Scenario
    x_r, y_r = [], []
    x_b, y_b = [], []
    x_o, y_o = [], []
    
    for p in sim_conf.particles:
        if p.color == 0: # Red
            x_r.append(p.x); y_r.append(L-1-p.y)
        elif p.color == 1: # Blue
            x_b.append(p.x); y_b.append(L-1-p.y)
        else: # Others
            x_o.append(p.x); y_o.append(L-1-p.y)
            
    ax2.scatter(x_o, y_o, c='lightgray', s=10, alpha=0.3, label='Neutral')
    ax2.scatter(x_r, y_r, c='red', s=30, edgecolors='white', label='Red (Side A)')
    ax2.scatter(x_b, y_b, c='blue', s=30, edgecolors='white', label='Blue (Side B)')
    
    ax2.set_xlim(-1, L); ax2.set_ylim(-1, L)
    ax2.set_title(f'Microstate of Frustration (T={STEPS})\n(Note: Separation/Domain Walls)')
    ax2.legend(loc='upper right')
    
    plt.suptitle("Figure 5: Semantic Frustration & The Cost of Hallucination")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_frustration_experiment()
