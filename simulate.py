"""
Interactive CartPole simulation with trained PPO model
- Stability features only when balanced, full power for recovery
"""

import pygame
import numpy as np
import torch
from cartpole_env import CartPole
from ppo_agent import PPOAgent
from config import PPOConfig


class AdaptiveActionSmoother:
    """Adaptive smoothing - heavy only when very stable"""
    def __init__(self):
        self.smoothed_action = 0.0
    
    def smooth(self, action, state):
        angle = abs(state[2])
        angle_vel = abs(state[3])
        
        # Only smooth heavily when VERY stable
        if angle < 3 * np.pi/180 and angle_vel < 0.05:
            # Very stable - heavy smoothing
            alpha = 0.15
        elif angle < 10 * np.pi/180 and angle_vel < 0.2:
            # Mostly stable - moderate smoothing
            alpha = 0.35
        else:
            # Recovering or unstable - minimal smoothing (fast response)
            alpha = 0.7
        
        self.smoothed_action = alpha * action + (1 - alpha) * self.smoothed_action
        return self.smoothed_action
    
    def reset(self):
        self.smoothed_action = 0.0


class SelectiveStabilityController:
    """Apply stability features ONLY when already balanced"""
    
    def apply_selective_damping(self, force, state):
        """Damping only when very upright - not during recovery"""
        x, x_dot, theta, theta_dot = state
        
        angle = abs(theta)
        
        # Only apply damping when pole is very upright
        if angle < 5 * np.pi/180:  # Less than 5°
            # Already balanced - add damping to prevent oscillation
            velocity_damping = -1.5 * x_dot
            angular_damping = -1.0 * theta_dot
            damped_force = force + velocity_damping + angular_damping
        else:
            # Recovering - NO damping, need full power
            damped_force = force
        
        return damped_force
    
    def apply_selective_reduction(self, force, state):
        """Reduce force only when nearly perfectly balanced"""
        x, x_dot, theta, theta_dot = state
        
        # Only reduce force when EXTREMELY well balanced
        if abs(theta) < 2 * np.pi/180 and abs(theta_dot) < 0.05 and abs(x) < 0.2:
            # Nearly perfect - gentle corrections only
            force *= 0.2
        elif abs(theta) < 5 * np.pi/180 and abs(theta_dot) < 0.1:
            # Very good - moderate reduction
            force *= 0.5
        # Otherwise: full force for recovery
        
        return force


# Initialize
pygame.init()
window_width, window_height = 600, 400
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("CartPole - Balanced Stability")
clock = pygame.time.Clock()

# Setup environment and agent
env = CartPole()
state = env.reset()

config = PPOConfig()
agent = PPOAgent(state_dim=4, action_dim=1, config=config)

# Load trained model
try:
    agent.load("models/ppo_cartpole_best.pth")
    print("✓ Loaded best model")
except:
    try:
        agent.load("models/ppo_cartpole.pth")
        print("✓ Loaded regular model")
    except:
        print("⚠ No trained model found - using random policy")

# Controllers
action_smoother = AdaptiveActionSmoother()
stability_controller = SelectiveStabilityController()

scale = 100
font = pygame.font.Font(None, 24)

dragging = False
running = True
steps = 0
use_stability = True

def draw(state, force, raw_force, mode="AI", control_mode=""):
    """Draw the CartPole state"""
    screen.fill((255, 255, 255))
    x, x_dot, theta, theta_dot = state

    # Draw ground
    pygame.draw.line(screen, (100, 100, 100), (0, window_height // 2), 
                    (window_width, window_height // 2), 2)

    # Draw boundaries
    left_bound = window_width // 2 - int(3.0 * scale)
    right_bound = window_width // 2 + int(3.0 * scale)
    pygame.draw.line(screen, (255, 0, 0), (left_bound, 0), (left_bound, window_height), 2)
    pygame.draw.line(screen, (255, 0, 0), (right_bound, 0), (right_bound, window_height), 2)

    # Draw center line
    center_x = window_width // 2
    pygame.draw.line(screen, (200, 200, 200), (center_x, 0), (center_x, window_height), 1)

    # Draw cart - color indicates control mode
    cart_y = window_height // 2
    cart_x = int(window_width // 2 + x * scale)
    if mode == "Manual":
        cart_color = (100, 100, 255)
    elif control_mode == "STABLE":
        cart_color = (0, 200, 0)  # Green - stable mode
    elif control_mode == "RECOVERY":
        cart_color = (255, 165, 0)  # Orange - recovery mode
    else:
        cart_color = (0, 0, 0)
    
    pygame.draw.rect(screen, cart_color, (cart_x - 25, cart_y - 15, 50, 30), 3)

    # Draw pole
    pole_x = cart_x + int(env.pole_length * 200 * np.sin(theta))
    pole_y = cart_y - 15 - int(env.pole_length * 200 * np.cos(theta))
    
    angle_deg = abs(theta * 180 / np.pi)
    if angle_deg < 5:
        pole_color = (0, 255, 0)
    elif angle_deg < 15:
        pole_color = (150, 200, 0)
    elif angle_deg < 45:
        pole_color = (200, 150, 0)
    else:
        pole_color = (255, 0, 0)
    
    pygame.draw.line(screen, pole_color, (cart_x, cart_y - 15), (pole_x, pole_y), 5)
    pygame.draw.circle(screen, (50, 50, 50), (pole_x, pole_y), 8)

    # Draw force indicator
    force_scale = 4
    force_x = int(cart_x + force * force_scale)
    if abs(force) > 0.5:
        pygame.draw.line(screen, (0, 0, 255), (cart_x, cart_y), 
                        (force_x, cart_y), 3)
        arrow_dir = 1 if force > 0 else -1
        pygame.draw.polygon(screen, (0, 0, 255), [
            (force_x, cart_y),
            (force_x - arrow_dir * 10, cart_y - 5),
            (force_x - arrow_dir * 10, cart_y + 5)
        ])

    # Display info
    info_texts = [
        f"Mode: {mode}",
        f"Control: {control_mode}" if control_mode else "",
        f"Steps: {steps}",
        f"Angle: {theta * 180 / np.pi:.1f}°",
        f"Position: {x:.2f}m",
        f"Velocity: {x_dot:.2f}m/s",
        f"Force: {force:.2f}N"
    ]
    
    y_offset = 10
    for text in info_texts:
        if text:
            text_surface = font.render(text, True, (0, 0, 0))
            screen.blit(text_surface, (10, y_offset))
            y_offset += 25

    # Instructions
    instructions = [
        "Click & Drag: Manual",
        "Release: AI control",
        "R: Reset",
        "D: Toggle stability",
        "ESC: Quit",
        f"Stability: {'ON' if use_stability else 'OFF'}"
    ]
    y_offset = window_height - 160
    for text in instructions:
        text_surface = font.render(text, True, (100, 100, 100))
        screen.blit(text_surface, (window_width - 200, y_offset))
        y_offset += 25

    pygame.display.flip()


# Main loop
print("\n" + "=" * 60)
print("Interactive CartPole Simulation - SELECTIVE STABILITY")
print("=" * 60)
print("Controls:")
print("  - Click and drag to control the cart")
print("  - Release to let the AI take over")
print("  - Press R to reset")
print("  - Press D to toggle stability features")
print("  - Press ESC to quit")
print("=" * 60)
print("Smart stability:")
print("  - GREEN cart = Stable mode (damping active)")
print("  - ORANGE cart = Recovery mode (full power)")
print("=" * 60 + "\n")

while running:
    force = 0.0
    raw_force = 0.0
    control_mode = ""
    mode = "Manual" if dragging else "AI"
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            dragging = True
            action_smoother.reset()
        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = False
            steps = 0
            action_smoother.reset()
            print(f"Released at: x={state[0]:.2f}m, theta={state[2]*180/np.pi:.1f}° - AI taking over...")
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                state = env.reset()
                steps = 0
                action_smoother.reset()
                print("Environment reset")
            elif event.key == pygame.K_d:
                use_stability = not use_stability
                print(f"Stability features: {'ENABLED' if use_stability else 'DISABLED'}")

    if dragging:
        # Manual control
        mouse_x, _ = pygame.mouse.get_pos()
        desired_x = (mouse_x - window_width / 2) / scale
        desired_x = np.clip(desired_x, -2.8, 2.8)
        
        error = desired_x - state[0]
        kp = 100.0
        kd = 20.0
        force = kp * error - kd * state[1]
        force = np.clip(force, -100.0, 100.0)
        
    else:
        # AI control
        action, _, _ = agent.select_action(state, deterministic=True)
        
        # INCREASED force scaling for recovery capability
        raw_force = action[0] * 6.0  # Increased from 3.0 to 6.0
        
        if use_stability:
            angle = abs(state[2])
            
            # Determine control mode
            if angle < 5 * np.pi/180:
                control_mode = "STABLE"
            else:
                control_mode = "RECOVERY"
            
            # ADAPTIVE SMOOTHING (minimal during recovery)
            force = action_smoother.smooth(raw_force, state)
            
            # SELECTIVE DAMPING (only when very upright)
            force = stability_controller.apply_selective_damping(force, state)
            
            # SELECTIVE FORCE REDUCTION (only when perfectly balanced)
            force = stability_controller.apply_selective_reduction(force, state)
        else:
            force = raw_force
            control_mode = "RAW"
        
        # Clip final force
        force = np.clip(force, -20.0, 20.0)  # Increased from 15 to 20
        
        # Boundary enforcement
        max_x = 3.0
        x = state[0]
        if x <= -max_x:
            force += 30.0 * (-max_x - x)
        elif x >= max_x:
            force += 30.0 * (max_x - x)

    # Step environment
    next_state, reward, done, info = env.step(force)
    state = next_state
    
    if not dragging:
        steps += 1

    draw(state, force, raw_force, mode, control_mode)
    clock.tick(50)

pygame.quit()
print("\nSimulation ended")