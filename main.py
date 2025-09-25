import pygame
import numpy as np

cart_mass = 1.0
pole_mass = 0.1
pole_length = 0.5
gravity = 9.8
dt = 0.01 # timestep
damping_cart = 0.1
damping_pole = 0.5

# state = [x, x_dot, theta, theta_dot]
state = np.array([0.0, 0.0, np.pi / 6, 0.0])

pygame.init()
window_width, window_height = 600, 400
screen = pygame.display.set_mode((window_width, window_height))
clock = pygame.time.Clock()

scale = 600 / (3.0 * 2) # for converting pixels to meters (600 pixels to 6 meters)
cart_limit = 3 # limit for the distance the cart can travel on either side in meters

dragging = False
running = True

# Note: pole was spinning out of control so i decided to try RK4 integration instead of eulers 
# RK4 equations can be found at https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

def derivatives(state, force):
    x, x_dot, theta, theta_dot = state

    # equations can be found at https://coneural.org/florian/papers/05_cart_pole.pdf
    denom = cart_mass + pole_mass
    theta_ddot = gravity * np.sin(theta) + np.cos(theta) * ((-force - pole_mass * pole_length * theta_dot ** 2 * np.sin(theta)) / denom) / \
    (pole_length * ((4.0 / 3.0) - (pole_mass * np.cos(theta) ** 2) / denom))
    x_ddot = (force + pole_mass * pole_length * (theta_dot ** 2 * np.sin(theta) - theta_ddot * np.cos(theta))) / denom

    x_ddot -= damping_cart * x_dot
    theta_ddot -= damping_pole * theta_dot
    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])


def rk4_step(state, force):
    k1 = derivatives(state, force)
    k2 = derivatives(state + dt/2 * k1, force)
    k3 = derivatives(state + dt/2 * k2, force)
    k4 = derivatives(state + dt * k3, force)

    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def draw(state):
    screen.fill((255, 255, 255))
    x, x_dot, theta, theta_dot = state

    # draw cart
    cart_y = window_height // 2
    cart_x = int(window_width // 2 + x * 100)
    pygame.draw.rect(screen, (0,0,0), (cart_x-25, cart_y-15, 50, 30), 2)

    # draw pole
    pole_x = cart_x + int(pole_length*200*np.sin(theta))
    pole_y = cart_y - int(pole_length*200*np.cos(theta))
    pygame.draw.line(screen, (200,0,0), (cart_x, cart_y), (pole_x, pole_y), 5)

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
        desired_x = max(-cart_limit, min(desired_x, cart_limit))

        # Using PD Controller to convert mouse movement to force
        # Info on PD Controller can be found at https://www.matthewpeterkelly.com/tutorials/pdControl/index.html
        kp = (np.pi * 2 * 3.0) ** 2
        kd = 2 * 1 * (np.pi * 3.0)

        force = kp * (desired_x - state[0]) - kd * state[1] # consider adding a max force in future
        state = rk4_step(state, force)
    else:
        # Later: replace with RL action
        pass

    state = rk4_step(state, force)
    draw(state)
    clock.tick(50)

pygame.quit()