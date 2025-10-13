import pygame
import numpy as np
from cartpole_env import CartPole
from ppo_train import ActorCritic
import torch

pygame.init()
window_width, window_height = 600, 400
screen = pygame.display.set_mode((window_width, window_height))
clock = pygame.time.Clock()

env = CartPole()
state = env.reset()

state_dim = 4
action_dim = 1

policy = ActorCritic(state_dim, action_dim)
policy.load_state_dict(torch.load("ppo_cartpole.pth"))
policy.eval()  # evaluation mode


scale = 600 / (3.0 * 2) # for converting pixels to meters (600 pixels to 6 meters)

dragging = False
running = True

def draw(state):
    screen.fill((255, 255, 255))
    x, x_dot, theta, theta_dot = state

    # draw cart
    cart_y = window_height // 2
    cart_x = int(window_width // 2 + x * 100)
    pygame.draw.rect(screen, (0,0,0), (cart_x-25, cart_y-15, 50, 30), 2)

    # draw pole
    pole_x = cart_x + int(env.pole_length*200*np.sin(theta))
    pole_y = cart_y - 15 - int(env.pole_length*200*np.cos(theta))
    pygame.draw.line(screen, (200,0,0), (cart_x, cart_y - 15), (pole_x, pole_y), 5)

    pygame.display.flip()

while running:
    force = 0.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = False


    if dragging:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        desired_x = (mouse_x - window_width/2) / scale

        kp = (np.pi * 2 * 3.0) ** 2
        kd = 2 * 1 * (np.pi * 3.0)
        pd_force = kp * (desired_x - state[0]) - kd * state[1]  # PD controller
        force = pd_force
    else:
        theta_threshold = 15 * np.pi / 180  # switch to PPO only when pole is within ±15 degrees
        if abs(state[2]) > theta_threshold:
            # Use PD controller for large angles
            kp = 30.0
            kd = 5.0
            force = kp * (0 - state[2]) - kd * state[3]
            alpha = min(abs(state[2]) / theta_threshold, 1.0)  # 0 to 1
            force = alpha * pd_force + (1 - alpha) * ppo_force

        else:
            # Use PPO
            state_tensor = torch.tensor(state, dtype=torch.float32)
            mu, std, _ = policy(state_tensor)
            ppo_force = mu.detach().numpy().item()
            max_force = 10.0
            force = np.clip(ppo_force * max_force, -max_force, max_force)


    
    max_x = 3.0
    x = state[0]
    if x <= -max_x:
            force += 50.0 * (-max_x - x)
    elif x >= max_x:
        force += 50.0 * (max_x - x)





    next_state, reward, done, info = env.step(force)
    state = next_state

    draw(state)
    clock.tick(50)

pygame.quit()