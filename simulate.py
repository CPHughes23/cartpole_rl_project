"""
Interactive CartPole simulation with trained PPO model
- Click and drag to manually control cart position
- Release to let the AI take over with smooth control
"""

import pygame
import numpy as np
import torch
from cartpole_env import CartPole
from ppo_agent import PPOAgent
from config import PPOConfig


class ActionSmoother:
    """Low-pass filter for smooth action transitions"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # Lower = more smoothing (0.1-0.5 good range)
        self.smoothed_action = 0.0
    
    def smooth(self, action):
        # Exponential moving average
        self.smoothed_action = self.alpha * action + (1 - self.alpha) * self.smoothed_action
        return self.smoothed_action
    
    def reset(self):
        self.smoothed_action = 0.0


# Initialize
pygame.init()
window_width, window_height = 600, 400
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("CartPole - PPO Agent (Smooth)")
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

# Action smoother for AI control
action_smoother = ActionSmoother(alpha=0.25)  # Adjust 0.1-0.5 for more/less smoothing

scale = 100  # pixels per meter
font = pygame.font.Font(None, 24)

dragging = False
running = True
total_reward = 0
steps = 0

def draw(state, force, mode="AI", smoothed=False):
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

    # Draw cart
    cart_y = window_height // 2
    cart_x = int(window_width // 2 + x * scale)
    cart_color = (100, 100, 255) if mode == "Manual" else (0, 150, 0) if smoothed else (0, 0, 0)
    pygame.draw.rect(screen, cart_color, (cart_x - 25, cart_y - 15, 50, 30), 3)

    # Draw pole
    pole_x = cart_x + int(env.pole_length * 200 * np.sin(theta))
    pole_y = cart_y - 15 - int(env.pole_length * 200 * np.cos(theta))
    
    # Color based on angle
    angle_deg = abs(theta * 180 / np.pi)
    if angle_deg < 10:
        pole_color = (0, 200, 0)  # Green - very upright
    elif angle_deg < 30:
        pole_color = (100, 150, 0)  # Yellow-green
    elif angle_deg < 60:
        pole_color = (200, 100, 0)  # Orange
    else:
        pole_color = (200, 0, 0)  # Red - very tilted
    
    pygame.draw.line(screen, pole_color, (cart_x, cart_y - 15), (pole_x, pole_y), 5)
    pygame.draw.circle(screen, (50, 50, 50), (pole_x, pole_y), 8)

    # Draw force indicator
    force_scale = 4
    force_x = int(cart_x + force * force_scale)
    if abs(force) > 0.5:
        pygame.draw.line(screen, (0, 0, 255), (cart_x, cart_y), 
                        (force_x, cart_y), 3)
        # Arrow head
        arrow_dir = 1 if force > 0 else -1
        pygame.draw.polygon(screen, (0, 0, 255), [
            (force_x, cart_y),
            (force_x - arrow_dir * 10, cart_y - 5),
            (force_x - arrow_dir * 10, cart_y + 5)
        ])

    # Display info
    info_texts = [
        f"Mode: {mode}{'(Smoothed)' if smoothed and mode=='AI' else ''}",
        f"Steps: {steps}",
        f"Angle: {theta * 180 / np.pi:.1f}°",
        f"Position: {x:.2f}m",
        f"Velocity: {x_dot:.2f}m/s",
        f"Force: {force:.2f}N"
    ]
    
    y_offset = 10
    for text in info_texts:
        text_surface = font.render(text, True, (0, 0, 0))
        screen.blit(text_surface, (10, y_offset))
        y_offset += 25

    # Instructions
    instructions = [
        "Click & Drag: Manual control",
        "Release: AI takes over",
        "R: Reset",
        "S: Toggle smoothing",
        "ESC: Quit"
    ]
    y_offset = window_height - 135
    for text in instructions:
        text_surface = font.render(text, True, (100, 100, 100))
        screen.blit(text_surface, (window_width - 210, y_offset))
        y_offset += 25

    pygame.display.flip()


# Main loop
print("\n" + "=" * 60)
print("Interactive CartPole Simulation - SMOOTH CONTROL")
print("=" * 60)
print("Controls:")
print("  - Click and drag to manually control the cart")
print("  - Release to let the AI take over")
print("  - Press R to reset")
print("  - Press S to toggle AI smoothing")
print("  - Press ESC to quit")
print("=" * 60)
print(f"Action smoothing: {'ENABLED' if action_smoother.alpha < 1.0 else 'DISABLED'}")
print("=" * 60 + "\n")

smoothing_enabled = True

while running:
    force = 0.0
    mode = "Manual" if dragging else "AI"
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            dragging = True
            action_smoother.reset()  # Reset smoother when user takes control
        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = False
            steps = 0
            action_smoother.reset()  # Reset smoother for clean AI takeover
            print(f"Released at: x={state[0]:.2f}m, theta={state[2]*180/np.pi:.1f}° - AI taking over...")
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                state = env.reset()
                steps = 0
                action_smoother.reset()
                print("Environment reset")
            elif event.key == pygame.K_s:
                smoothing_enabled = not smoothing_enabled
                action_smoother.alpha = 0.25 if smoothing_enabled else 1.0
                print(f"Smoothing: {'ENABLED' if smoothing_enabled else 'DISABLED'}")

    if dragging:
        # Manual control - strong force to follow mouse
        mouse_x, _ = pygame.mouse.get_pos()
        desired_x = (mouse_x - window_width / 2) / scale
        desired_x = np.clip(desired_x, -2.8, 2.8)
        
        error = desired_x - state[0]
        kp = 100.0
        kd = 20.0
        force = kp * error - kd * state[1]
        force = np.clip(force, -100.0, 100.0)
        
    else:
        # AI control with smoothing
        action, _, _ = agent.select_action(state, deterministic=True)
        
        # REDUCED FORCE SCALING (was 10.0, now 5.0)
        raw_force = action[0] * 5.0
        
        # SMOOTH THE ACTION
        if smoothing_enabled:
            force = action_smoother.smooth(raw_force)
        else:
            force = raw_force
        
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

    draw(state, force, mode, smoothed=smoothing_enabled)
    clock.tick(50)

pygame.quit()
print("\nSimulation ended")