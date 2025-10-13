import numpy as np

class CartPole:
    def __init__(self):
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.pole_length = 0.5
        self.gravity = 9.8
        self.dt = 0.02 # timestep
        self.damping_cart = 0.1
        self.damping_pole = 0.5
        

        self.state = None
        self.steps = 0
        self.max_steps = 500

    def reset(self):
        self.state = np.random.uniform(low=-0.45, high=0.45, size=(4,)) # resets to a semi-random position so that training has variance
        self.steps = 0
        return self.state
    
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        attributes = np.array([self.cart_mass, self.pole_mass, self.pole_length, self.gravity, self.damping_cart, self.damping_pole])
        force = float(action)

        # Note: pole was spinning out of control so I decided to try RK4 integration instead of eulers 
        # RK4 equations can be found at https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        k1 = self.__derivatives(self.state, force, attributes)
        k2 = self.__derivatives(self.state + self.dt/2 * k1, force, attributes)
        k3 = self.__derivatives(self.state + self.dt/2 * k2, force, attributes)
        k4 = self.__derivatives(self.state + self.dt * k3, force, attributes)
        self.state = self.state + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        self.steps += 1



        terminated = bool(
            x < -3 or x > 3 or # cart outside of windows bounds
            theta < -12 * np.pi/180 or theta > 12 * np.pi/180 or # pole has fallen over
            self.steps >= self.max_steps # ran out of steps
        )
        max_x = 3.0
        max_v = 5.0
        self.state[0] = np.clip(self.state[0], -max_x, max_x)  # x
        self.state[1] = np.clip(self.state[1], -max_v, max_v)  # x_dot


        desired_x = 0.0
        if not terminated:
            reward = 1.0 - 0.1 * abs(theta) - 0.01 * abs(theta_dot) - 0.05 * abs(x - desired_x)
            reward -= 0.1 * max(0, abs(x) - 2.5)
            if x < -max_x or x > max_x:
                reward = -10.0
        else:
            reward = 0
        info = {"x":x, "theta":theta}
        return self.state, reward, terminated, info
        
    def __derivatives(self, state, force, attributes):
        x, x_dot, theta, theta_dot = state

        cart_mass = attributes[0]
        pole_mass = attributes[1]
        pole_length = attributes[2]
        gravity = attributes[3]
        damping_cart = attributes[4]
        damping_pole = attributes[5]
        
        # equations can be found at https://coneural.org/florian/papers/05_cart_pole.pdf
        total_mass = cart_mass + pole_mass
        theta_ddot = gravity * np.sin(theta) + np.cos(theta) * ((-force - pole_mass * pole_length * theta_dot ** 2 * np.sin(theta)) / total_mass) / \
        (pole_length * ((4.0 / 3.0) - (pole_mass * np.cos(theta) ** 2) / total_mass))
        x_ddot = (force + pole_mass * pole_length * (theta_dot ** 2 * np.sin(theta) - theta_ddot * np.cos(theta))) / total_mass

        x_ddot -= damping_cart * x_dot
        theta_ddot -= damping_pole * theta_dot

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])