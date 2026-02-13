"""
Interactive CartPole simulation with trained PPO model
- Click and drag to manually control cart position
- Release to let the AI take over
"""

import pygame
import numpy as np
import torch
from cartpole_env import CartPole
from ppo_agent import PPOAgent
from config import PPOConfig

# Initialize
pygame.init()
window_width, window_height = 600, 400
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("CartPole - PPO Agent")
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

scale = 600 / (3.0 * 2)  # Convert pixels to meters
font = pygame.font.Font(None, 24)

dragging = False
running = True
total_reward = 0
steps = 0

def draw(state, force, reward, mode="AI"):
    """Draw the CartPole state"""
    screen.fill((255, 255, 255))
    x, x_dot, theta, theta_dot = state

    # Draw ground
    pygame.draw.line(screen, (100, 100, 100), (0, window_height // 2), 
                    (window_width, window_height // 2), 2)

    # Draw boundaries
    left_bound = window_width // 2 - int(3.0 * 100)
    right_bound = window_width // 2 + int(3.0 * 100)
    pygame.draw.line(screen, (255, 0, 0), (left_bound, 0), (left_bound, window_height), 2)
    pygame.draw.line(screen, (255, 0, 0), (right_bound, 0), (right_bound, window_height), 2)

    # Draw cart
    cart_y = window_height // 2
    cart_x = int(window_width // 2 + x * 100)
    cart_color = (100, 100, 255) if mode == "Manual" else (0, 0, 0)
    pygame.draw.rect(screen, cart_color, (cart_x - 25, cart_y - 15, 50, 30), 3)

    # Draw pole
    pole_x = cart_x + int(env.pole_length * 200 * np.sin(theta))
    pole_y = cart_y - 15 - int(env.pole_length * 200 * np.cos(theta))
    pole_color = (200, 0, 0) if abs(theta) > 10 * np.pi / 180 else (0, 150, 0)
    pygame.draw.line(screen, pole_color, (cart_x, cart_y - 15), (pole_x, pole_y), 5)
    pygame.draw.circle(screen, (50, 50, 50), (pole_x, pole_y), 8)

    # Draw force indicator
    force_scale = 5
    force_x = int(cart_x + force * force_scale)
    if abs(force) > 0.1:
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
        f"Mode: {mode}",
        f"Steps: {steps}",
        f"Reward: {total_reward:.2f}",
        f"Angle: {theta * 180 / np.pi:.1f}°",
        f"Position: {x:.2f}m",
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
        "Release: AI control",
        "R: Reset",
        "ESC: Quit"
    ]
    y_offset = window_height - 110
    for text in instructions:
        text_surface = font.render(text, True, (100, 100, 100))
        screen.blit(text_surface, (window_width - 200, y_offset))
        y_offset += 25

    pygame.display.flip()


# Main loop
print("\n" + "=" * 60)
print("Interactive CartPole Simulation")
print("=" * 60)
print("Controls:")
print("  - Click and drag to manually control the cart")
print("  - Release to let the AI take over")
print("  - Press R to reset")
print("  - Press ESC to quit")
print("=" * 60 + "\n")

while running:
    force = 0.0
    mode = "Manual" if dragging else "AI"
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                state = env.reset()
                total_reward = 0
                steps = 0
                print("Environment reset")

    if dragging:
        # Manual control with PD controller
        mouse_x, _ = pygame.mouse.get_pos()
        desired_x = (mouse_x - window_width / 2) / 100
        
        kp = (np.pi * 2 * 3.0) ** 2
        kd = 2 * 1 * (np.pi * 3.0)
        force = kp * (desired_x - state[0]) - kd * state[1]
    else:
        # AI control
        action, _, _ = agent.select_action(state, deterministic=True)
        force = action[0] * 10.0  # Scale to appropriate force range

    # Boundary enforcement
    max_x = 3.0
    x = state[0]
    if x <= -max_x:
        force += 50.0 * (-max_x - x)
    elif x >= max_x:
        force += 50.0 * (max_x - x)

    # Step environment
    next_state, reward, done, info = env.step(force)
    state = next_state
    total_reward += reward
    steps += 1

    if done:
        print(f"Episode finished - Steps: {steps}, Total Reward: {total_reward:.2f}")
        state = env.reset()
        total_reward = 0
        steps = 0

    draw(state, force, reward, mode)
    clock.tick(50)

pygame.quit()
print("\nSimulation ended")