import pygame
import numpy as np
from cartpole_env import CartPole

pygame.init()
window_width, window_height = 600, 400
screen = pygame.display.set_mode((window_width, window_height))
clock = pygame.time.Clock()

env = CartPole()
state = env.reset()

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

        # Using PD Controller to convert mouse movement to force
        # Info on PD Controller can be found at https://www.matthewpeterkelly.com/tutorials/pdControl/index.html
        kp = (np.pi * 2 * 3.0) ** 2
        kd = 2 * 1 * (np.pi * 3.0)

        force = kp * (desired_x - state[0]) - kd * state[1] # consider adding a max force in future
        env.step(force)
        state = env.state
    else:
        # Later: replace with RL action
        pass

    env.step(force)
    state = env.state
    draw(state)
    clock.tick(50)

pygame.quit()