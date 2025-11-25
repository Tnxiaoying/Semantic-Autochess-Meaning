import numpy as np
import matplotlib.pyplot as plt
import random
import copy

# 复用主程序的核心类，但为了独立运行，这里简化重写核心逻辑
# ==========================================
# Core Logic
# ==========================================
L = 40
N_COLORS = 4

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

def generate_scenario_simple(n_particles, mode='random'):
    sim = SemanticGrid(L, L)
    particles_per_color = n_particles // N_COLORS
    colors = []
    for c in range(N_COLORS): colors.extend([c] * particles_per_color)
    
    all_coords = [(x, y) for x in range(L) for y in range(L)]
    
    if mode == 'random':
        positions = random.sample(all_coords, n_particles)
        random.shuffle(colors)
    else: # clustered
        positions = []
        centers = [(L//4, L//4), (3*L//4, L//4), (L//4, 3*L//4), (3*L//4, 3*L//4)]
        occupied = set()
        for c_idx in range(N_COLORS):
            ct = 0
            while ct < particles_per_color:
                rx = int(random.gauss(centers[c_idx][0], 3.0))
                ry = int(random.gauss(centers[c_idx][1], 3.0))
                rx, ry = max(0, min(L-1, rx)), max(0, min(L-1, ry))
                if (rx, ry) not in occupied:
                    occupied.add((rx, ry))
                    positions.append((rx, ry))
                    ct += 1
    
    for i in range(n_particles):
        sim.add_particle(Particle(i, colors[i], positions[i][0], positions[i][1]))
    return sim

def run_sim(sim, steps=200, q_syn=0.01, q_col=1.0):
    total_q = 0
    for _ in range(steps):
        particles = list(sim.particles)
        random.shuffle(particles)
        for p in particles:
            dx, dy = random.choice([(-1,0), (1,0), (0,-1), (0,1)])
            nx, ny = p.x + dx, p.y + dy
            if not (0 <= nx < L and 0 <= ny < L):
                total_q += q_col
                continue
            
            target = sim.get_particle_at(nx, ny)
            if target is None:
                sim.move_particle(p, nx, ny)
                total_q += 0.1
            elif target.color != p.color:
                total_q += q_col
            else:
                sim.swap_particles(p, target)
                total_q += q_syn
    return total_q

# ==========================================
# Experiments
# ==========================================

def run_sensitivity_analysis():
    print("Running Sensitivity Analysis (Figure 4)...")
    plt.rcParams.update({'font.size': 10})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Experiment A: Density Sweep ---
    print("1. Density Sweep...")
    densities = np.linspace(50, 1200, 10, dtype=int) # From sparse to crowded
    savings = []
    
    for n in densities:
        # Run Random
        sim_rnd = generate_scenario_simple(n, 'random')
        q_rnd = run_sim(sim_rnd)
        
        # Run Clustered
        sim_clu = generate_scenario_simple(n, 'clustered')
        q_clu = run_sim(sim_clu)
        
        # Calculate Saving %
        saving_pct = (q_rnd - q_clu) / q_rnd * 100
        savings.append(saving_pct)
        
    ax1.plot(densities, savings, 'o-', color='tab:blue', lw=2)
    ax1.set_xlabel('Number of Particles (N)')
    ax1.set_ylabel('Energy Saving from Structure (%)')
    ax1.set_title('Robustness Check: Density Dependence\n(Where does Structure matter most?)')
    ax1.grid(True, alpha=0.3)
    ax1.axvspan(300, 600, color='yellow', alpha=0.1, label='Critical Region')
    ax1.legend()

    # --- Experiment B: Cost Ratio Sweep ---
    print("2. Physics Cost Sweep...")
    synergy_costs = np.linspace(0.0, 0.9, 10) # From free to almost expensive
    linear_fits = [] # We want to see if Q_rnd vs Q_clu stays robust
    
    # Fixed N = 400
    q_r_list = []
    q_c_list = []
    
    for cost in synergy_costs:
        sim_rnd = generate_scenario_simple(400, 'random')
        q_rnd = run_sim(sim_rnd, q_syn=cost)
        
        sim_clu = generate_scenario_simple(400, 'clustered')
        q_clu = run_sim(sim_clu, q_syn=cost)
        
        q_r_list.append(q_rnd)
        q_c_list.append(q_clu)
        
    ax2.plot(synergy_costs, q_r_list, 'r--', label='High Entropy (Noise)')
    ax2.plot(synergy_costs, q_c_list, 'g-', label='Low Entropy (Structure)')
    
    # Calculate gap closing
    ax2.set_xlabel('Synergy Cost ($Q_{synergy}$)')
    ax2.set_ylabel('Total Heat ($Q_{total}$)')
    ax2.set_title('Robustness Check: Cost Sensitivity\n(Does benefit persist with higher costs?)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Figure 4: Universality & Robustness Checks")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_sensitivity_analysis()
