import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec

# Vehicle dynamics constants
drag_coefficient = 0.1  # Air resistance coefficient
throttle_effectiveness = 5.0  # How much acceleration per unit throttle
target_speed = 25  # Target speed (km/h or m/s)
initial_speed = 0  # Starting speed

def next_speed(throttle, current_speed, dt):
    """
    Vehicle speed dynamics: dv/dt = throttle_effectiveness * throttle - drag_coefficient * v
    This models acceleration from throttle minus drag proportional to speed
    """
    acceleration = throttle_effectiveness * throttle - drag_coefficient * current_speed
    return current_speed + acceleration * dt

def simulate_speed_control(controller, num_steps=100, dt=0.1):
    """Simulate vehicle speed control with given controller"""
    speed = initial_speed
    speed_history = [speed]
    time_history = [0]
    
    for step in range(num_steps):
        # Get control input from controller
        throttle = controller.get_control(speed, dt)
        # Limit throttle to realistic range [0, 1]
        throttle = np.clip(throttle, 0, 1)
        
        # Update vehicle speed
        speed = next_speed(throttle, speed, dt)
        speed_history.append(speed)
        time_history.append((step + 1) * dt)
    
    return np.array(time_history), np.array(speed_history)

class PController:
    """Proportional Controller"""
    def __init__(self, Kp, set_point):
        self.Kp = Kp
        self.set_point = set_point
    
    def get_control(self, measurement, dt):
        error = self.set_point - measurement
        return self.Kp * error

class PIController:
    """Proportional-Integral Controller"""
    def __init__(self, Kp, Ki, set_point):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = set_point
        self.integral_term = 0
    
    def get_control(self, measurement, dt):
        error = self.set_point - measurement
        self.integral_term += error * self.Ki * dt
        return self.Kp * error + self.integral_term

class PIDController:
    """Proportional-Integral-Derivative Controller"""
    def __init__(self, Kp, Ki, Kd, set_point):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = set_point
        self.integral_term = 0
        self.last_error = None
    
    def get_control(self, measurement, dt):
        error = self.set_point - measurement
        self.integral_term += error * self.Ki * dt
        
        derivative_term = 0
        if self.last_error is not None:
            derivative_term = (error - self.last_error) / dt * self.Kd
        
        self.last_error = error
        return self.Kp * error + self.integral_term + derivative_term

def plot_all_controllers():
    """Plot comparison of all three controller types"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Vehicle Speed Control: P vs PI vs PID Controllers', fontsize=16)
    
    # Controller parameters (tuned for good performance)
    controllers = {
        'P Controller': PController(Kp=0.3, set_point=target_speed),
        'PI Controller': PIController(Kp=0.3, Ki=0.01, set_point=target_speed),
        'PID Controller': PIDController(Kp=0.3, Ki=0.01, Kd=0.01, set_point=target_speed)
    }
    
    colors = ['blue', 'green', 'red']
    positions = [(0, 0), (0, 1), (1, 0)]
    
    for i, (name, controller) in enumerate(controllers.items()):
        ax = axes[positions[i][0], positions[i][1]]
        
        # Simulate and plot
        time, speed = simulate_speed_control(controller, num_steps=150)
        
        ax.plot(time, speed, color=colors[i], linewidth=2, label=f'Actual Speed')
        ax.axhline(y=target_speed, color='black', linestyle='--', alpha=0.7, label=f'Target Speed ({target_speed})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 35)
    
    # Comparison plot
    ax = axes[1, 1]
    for i, (name, controller) in enumerate(controllers.items()):
        # Reset controllers for fair comparison
        if hasattr(controller, 'integral_term'):
            controller.integral_term = 0
        if hasattr(controller, 'last_error'):
            controller.last_error = None
            
        time, speed = simulate_speed_control(controller, num_steps=150)
        ax.plot(time, speed, color=colors[i], linewidth=2, label=name)
    
    ax.axhline(y=target_speed, color='black', linestyle='--', alpha=0.7, label=f'Target Speed ({target_speed})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title('Comparison of All Controllers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 35)
    
    plt.tight_layout()
    plt.show()

def interactive_pid_tuning():
    """Interactive PID parameter tuning with sliders"""
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Main plot
    ax_main = fig.add_subplot(gs[0])
    
    # Slider area
    ax_sliders = fig.add_subplot(gs[1])
    ax_sliders.set_visible(False)
    
    # Initial parameters
    init_kp = 0.12
    init_ki = 0.08
    init_kd = 0.02
    
    # Create sliders
    ax_kp = plt.axes([0.2, 0.25, 0.5, 0.03])
    ax_ki = plt.axes([0.2, 0.20, 0.5, 0.03])
    ax_kd = plt.axes([0.2, 0.15, 0.5, 0.03])
    
    slider_kp = Slider(ax_kp, 'Kp', 0.0, 2, valinit=init_kp)
    slider_ki = Slider(ax_ki, 'Ki', 0.0, 0.3, valinit=init_ki)
    slider_kd = Slider(ax_kd, 'Kd', 0.0, 0.1, valinit=init_kd)
    
    def update_plot():
        ax_main.clear()
        
        # Create new PID controller with current slider values
        controller = PIDController(
            Kp=slider_kp.val,
            Ki=slider_ki.val,
            Kd=slider_kd.val,
            set_point=target_speed
        )
        
        # Simulate and plot
        time, speed = simulate_speed_control(controller, num_steps=150)
        
        ax_main.plot(time, speed, 'b-', linewidth=2, label='Actual Speed')
        ax_main.axhline(y=target_speed, color='red', linestyle='--', alpha=0.7, 
                       label=f'Target Speed ({target_speed})')
        ax_main.set_xlabel('Time (s)')
        ax_main.set_ylabel('Speed (km/h)')
        ax_main.set_title(f'Interactive PID Tuning - Kp={slider_kp.val:.3f}, Ki={slider_ki.val:.3f}, Kd={slider_kd.val:.3f}')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        ax_main.set_ylim(0, 35)
        plt.draw()
    
    # Update function for sliders
    def on_slider_change(val):
        update_plot()
    
    slider_kp.on_changed(on_slider_change)
    slider_ki.on_changed(on_slider_change)
    slider_kd.on_changed(on_slider_change)
    
    # Initial plot
    update_plot()
    
    plt.show()

def demonstrate_steady_state_error():
    """Demonstrate the steady-state error problem with P controller"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Steady-State Error Demonstration', fontsize=16)
    
    # P Controller with different gains
    kp_values = [0.1, 0.15, 0.2]
    colors = ['blue', 'green', 'red']
    
    for i, kp in enumerate(kp_values):
        controller = PController(Kp=kp, set_point=target_speed)
        time, speed = simulate_speed_control(controller, num_steps=200)
        ax1.plot(time, speed, color=colors[i], linewidth=2, label=f'Kp = {kp}')
    
    ax1.axhline(y=target_speed, color='black', linestyle='--', alpha=0.7, 
               label=f'Target Speed ({target_speed})')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Speed (km/h)')
    ax1.set_title('P Controller - Different Gains')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 30)
    
    # Compare P vs PI
    p_controller = PController(Kp=0.15, set_point=target_speed)
    pi_controller = PIController(Kp=0.12, Ki=0.08, set_point=target_speed)
    
    time_p, speed_p = simulate_speed_control(p_controller, num_steps=200)
    time_pi, speed_pi = simulate_speed_control(pi_controller, num_steps=200)
    
    ax2.plot(time_p, speed_p, 'b-', linewidth=2, label='P Controller')
    ax2.plot(time_pi, speed_pi, 'g-', linewidth=2, label='PI Controller')
    ax2.axhline(y=target_speed, color='black', linestyle='--', alpha=0.7, 
               label=f'Target Speed ({target_speed})')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Speed (km/h)')
    ax2.set_title('P vs PI Controller')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 30)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run all demonstrations"""
    print("Vehicle Speed Control with PID Controllers")
    print("==========================================")
    print(f"Target Speed: {target_speed} km/h")
    print(f"Initial Speed: {initial_speed} km/h")
    print()
    
    print("1. Comparison of P, PI, and PID Controllers")
    plot_all_controllers()
    
    print("\n2. Steady-State Error Demonstration")
    demonstrate_steady_state_error()
    
    print("\n3. Interactive PID Tuning")
    print("Use the sliders to adjust Kp, Ki, and Kd values")
    interactive_pid_tuning()
    
    print("\nDemonstration complete!")
    print("\nKey observations:")
    print("- P Controller: Fast response but steady-state error")
    print("- PI Controller: Eliminates steady-state error but may overshoot")
    print("- PID Controller: Best overall performance with proper tuning")

if __name__ == "__main__":
    main() 