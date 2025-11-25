import numpy as np
import matplotlib.pyplot as plt
import random
import zlib
import copy

# ==========================================
# 1. Configuration
# ==========================================
L = 40              # Grid Size (LxL)
N_PARTICLES = 400   # Total Particles
N_COLORS = 4        # Number of Colors
STEPS = 500         # Simulation Steps

# Thermodynamic Costs
Q_MOVE = 0.1        # Free Move Heat
Q_COLLISION = 1.0   # Hard Scattering Heat (Noise/Mismatch)
Q_SYNERGY = 0.01    # Synergy Bond Heat (Structure/Match)

# Visualization Colors
COLOR_MAP = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'orange' 
}

# ==========================================
# 2. Core Classes
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
        self.grid = np.full((width, height), -1, dtype=int) # -1 means empty
        self.particles = []
        
    def add_particle(self, particle):
        self.particles.append(particle)
        self.grid[particle.x, particle.y] = particle.id

    def get_particle_at(self, x, y):
        p_id = self.grid[x, y]
        if p_id == -1:
            return None
        # Simple linear search
        for p in self.particles:
            if p.id == p_id:
                return p
        return None

    def move_particle(self, p, new_x, new_y):
        # Clear old position
        self.grid[p.x, p.y] = -1
        # Update new position
        p.x, p.y = new_x, new_y
        self.grid[new_x, new_y] = p.id
        
    def swap_particles(self, p1, p2):
        # Swap IDs on grid
        self.grid[p1.x, p1.y] = p2.id
        self.grid[p2.x, p2.y] = p1.id
        # Swap coordinates
        p1.x, p1.y, p2.x, p2.y = p2.x, p2.y, p1.x, p1.y

# ==========================================
# 3. Initialization & Metrics
# ==========================================

def generate_scenario(mode='random', cluster_spread=3.0):
    """
    Generate Scenario.
    Strict Histogram Constraint: Exact same count for each color (N/4).
    """
    sim = SemanticGrid(L, L)
    particles_per_color = N_PARTICLES // N_COLORS
    
    # 1. Prepare Particle Pool (Strict Constraint)
    particle_colors = []
    for c in range(N_COLORS):
        particle_colors.extend([c] * particles_per_color)
    
    # 2. Determine Positions
    positions = []
    all_coords = [(x, y) for x in range(L) for y in range(L)]
    
    if mode == 'random':
        # Scenario A: Max Entropy (Random Scatter)
        positions = random.sample(all_coords, N_PARTICLES)
        random.shuffle(particle_colors) # Shuffle colors
        
    elif mode == 'clustered':
        # Scenario B: Low Entropy (Structure/Clustered)
        # Define centers for N_COLORS
        quadrant_centers = [
            (L//4, L//4), (3*L//4, L//4),
            (L//4, 3*L//4), (3*L//4, 3*L//4)
        ]
        
        # Assign positions for each color
        occupied = set()
        
        for color_idx in range(N_COLORS):
            center = quadrant_centers[color_idx % len(quadrant_centers)]
            count = 0
            while count < particles_per_color:
                # Gaussian sampling around center
                rx = int(random.gauss(center[0], cluster_spread))
                ry = int(random.gauss(center[1], cluster_spread))
                
                # Clip to boundaries
                rx = max(0, min(L-1, rx))
                ry = max(0, min(L-1, ry))
                
                pos = (rx, ry)
                if pos not in occupied:
                    occupied.add(pos)
                    positions.append(pos)
                    count += 1
        
        # Rebuild color list to match position generation order
        particle_colors = []
        for c in range(N_COLORS):
            particle_colors.extend([c] * particles_per_color)

    # 3. Create Entities
    for i in range(N_PARTICLES):
        p = Particle(i, particle_colors[i], positions[i][0], positions[i][1])
        sim.add_particle(p)
        
    return sim

def calculate_semantic_entropy_proxy(sim):
    """
    Calculate Semantic Entropy Proxy using Lempel-Ziv Ratio.
    """
    grid_bytes = bytearray()
    for x in range(L):
        for y in range(L):
            val = sim.grid[x, y]
            if val == -1:
                grid_bytes.append(255) # Empty
            else:
                p = sim.get_particle_at(x, y)
                if p:
                    grid_bytes.append(p.color)
                else:
                    grid_bytes.append(255)
    
    compressed = zlib.compress(grid_bytes)
    # Ratio > 1 means hard to compress/high entropy
    ratio = len(compressed) / len(grid_bytes)
    return ratio

# ==========================================
# 4. Physics Engine
# ==========================================

def run_simulation_step(sim):
    """
    Execute one simulation step.
    Returns: (step_heat, hard_collisions, total_attempts)
    """
    step_heat = 0.0
    hard_collisions = 0
    total_attempts = 0
    
    # Random update order
    particles = list(sim.particles)
    random.shuffle(particles)
    
    for p in particles:
        total_attempts += 1
        
        # 1. Attempt Move (Random Neighbor)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = random.choice(moves)
        target_x, target_y = p.x + dx, p.y + dy
        
        # Boundary Check
        if not (0 <= target_x < L and 0 <= target_y < L):
            step_heat += Q_COLLISION # Wall bounce treated as damping
            hard_collisions += 1
            continue
            
        target_p = sim.get_particle_at(target_x, target_y)
        
        if target_p is None:
            # Case 1: Target Empty -> Free Move
            sim.move_particle(p, target_x, target_y)
            step_heat += Q_MOVE
            
        elif target_p.color != p.color:
            # Case 2: Different Color -> Hard Scattering (High Heat)
            # Move fails
            step_heat += Q_COLLISION
            hard_collisions += 1
            
        elif target_p.color == p.color:
            # Case 3: Same Color -> Synergy/Tunneling (Low Heat)
            # Swap positions
            sim.swap_particles(p, target_p)
            step_heat += Q_SYNERGY 
            
    return step_heat, hard_collisions, total_attempts

# ==========================================
# 5. Main Experiment & Visualization
# ==========================================

def run_experiment():
    print("Initializing The Information Experiments (V2.1 - English)...")
    
    # --- 1. Define Scenarios ---
    scenarios = [
        {"name": "Scenario A: High Entropy (Noise)", "mode": "random", "spread": None},
        {"name": "Scenario B: Low Entropy (Structure)", "mode": "clustered", "spread": 4.0},
    ]
    
    results = {}
    
    # Containers for Scatter Plot
    scatter_entropy = []
    scatter_heat = []

    # --- 2. Run Main Comparison ---
    for sc in scenarios:
        print(f"Running {sc['name']}...")
        sim = generate_scenario(mode=sc['mode'], cluster_spread=sc['spread'])
        
        initial_entropy = calculate_semantic_entropy_proxy(sim)
        
        history_Q = []
        history_rho = []
        cumulative_Q = 0
        
        for t in range(STEPS):
            heat, cols, attempts = run_simulation_step(sim)
            cumulative_Q += heat
            rho = cols / attempts if attempts > 0 else 0
            
            history_Q.append(cumulative_Q)
            history_rho.append(rho)
        
        results[sc['name']] = {
            "entropy": initial_entropy,
            "Q_total": history_Q,
            "rho": history_rho
        }
        
        scatter_entropy.append(initial_entropy)
        scatter_heat.append(cumulative_Q)

    # --- 3. Run Correlation Samples ---
    print("Running intermediate entropy scenarios for correlation plot...")
    spreads = [2.0, 3.0, 5.0, 8.0, 15.0] 
    for spr in spreads:
        sim = generate_scenario(mode='clustered', cluster_spread=spr)
        ent = calculate_semantic_entropy_proxy(sim)
        cum_Q = 0
        for t in range(STEPS):
            heat, _, _ = run_simulation_step(sim)
            cum_Q += heat
        scatter_entropy.append(ent)
        scatter_heat.append(cum_Q)
        
    # Additional random samples
    for _ in range(3):
        sim = generate_scenario(mode='random')
        ent = calculate_semantic_entropy_proxy(sim)
        cum_Q = 0
        for t in range(STEPS):
            heat, _, _ = run_simulation_step(sim)
            cum_Q += heat
        scatter_entropy.append(ent)
        scatter_heat.append(cum_Q)

    # ==========================================
    # 6. Plotting
    # ==========================================
    
    # Use default fonts to avoid encoding issues
    plt.rcParams.update({'font.size': 10})
    
    # --- Figure 1: The World (Initial States) ---
    fig1, axes = plt.subplots(1, 2, figsize=(12, 6))
    modes = [("random", None, "Scenario A: Max Entropy (Noise)"), 
             ("clustered", 4.0, "Scenario B: Low Entropy (Structure)")]
    
    for i, (m, s, title) in enumerate(modes):
        sim_viz = generate_scenario(mode=m, cluster_spread=s)
        S_val = calculate_semantic_entropy_proxy(sim_viz)
        
        x_vals, y_vals, c_vals = [], [], []
        for p in sim_viz.particles:
            x_vals.append(p.x)
            y_vals.append(L - 1 - p.y)
            c_vals.append(COLOR_MAP[p.color])
            
        axes[i].scatter(x_vals, y_vals, c=c_vals, s=20, alpha=0.8)
        axes[i].set_xlim(-1, L)
        axes[i].set_ylim(-1, L)
        axes[i].set_title(f"{title}\nProxy Entropy: {S_val:.4f}")
        axes[i].grid(True, linestyle='--', alpha=0.3)
    plt.suptitle("Figure 1: Initial World States (Strict Histogram Constraint)")
    plt.tight_layout()
    
    # --- Figure 2: The Thermodynamics (Time Series) ---
    fig2, ax1 = plt.subplots(figsize=(10, 6))
    
    color_A = 'tab:red'
    color_B = 'tab:green'
    name_A = scenarios[0]['name']
    name_B = scenarios[1]['name']
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Cumulative Heat (Q)', color='black')
    l1, = ax1.plot(results[name_A]['Q_total'], color=color_A, lw=2, label=f'{name_A} (Heat)')
    l2, = ax1.plot(results[name_B]['Q_total'], color=color_B, lw=2, label=f'{name_B} (Heat)')
    ax1.tick_params(axis='y', labelcolor='black')
    
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Effective Jamming Rate (rho)', color='blue')
    
    # Simple Smoothing
    def smooth(y, box_pts=20):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    l3, = ax2.plot(smooth(results[name_A]['rho']), color=color_A, linestyle=':', alpha=0.6, label=f'{name_A} (Jamming)')
    l4, = ax2.plot(smooth(results[name_B]['rho']), color=color_B, linestyle=':', alpha=0.6, label=f'{name_B} (Jamming)')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 1.0)
    
    lines = [l1, l2, l3, l4]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper left')
    plt.title('Figure 2: Thermodynamics\nStructure Reduces Dissipation')

    # --- Figure 3: The Correlation (Scatter) ---
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    ax3.scatter(scatter_entropy, scatter_heat, c='purple', alpha=0.6, edgecolors='k')
    
    # Highlight Main A and B
    ax3.scatter([scatter_entropy[0]], [scatter_heat[0]], c='red', s=150, label='Scenario A', marker='*')
    ax3.scatter([scatter_entropy[1]], [scatter_heat[1]], c='green', s=150, label='Scenario B', marker='*')
    
    # Trend line
    z = np.polyfit(scatter_entropy, scatter_heat, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(scatter_entropy), max(scatter_entropy), 100)
    ax3.plot(x_trend, p(x_trend), "k--", alpha=0.5, label='Trend')
    
    ax3.set_xlabel('Semantic Entropy Proxy (LZ Ratio)')
    ax3.set_ylabel('Final Total Heat (Q)')
    ax3.set_title('Figure 3: The Information-Energy Equivalence\n(Understanding is Cooling)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.show()

if __name__ == "__main__":
    run_experiment()
