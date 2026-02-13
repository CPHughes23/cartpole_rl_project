import numpy as np

class CartPole:
    def __init__(self):
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.pole_length = 0.5
        self.gravity = 9.8
        self.dt = 0.02  # timestep
        self.damping_cart = 0.1
        self.damping_pole = 0.5
        
        self.state = None
        self.steps = 0
        self.max_steps = 1000  # Increased for full-range training
        self.edge_timer = 0

    def reset(self, recovery_training=False):
        if recovery_training:
            # FULL RANGE training - any angle, any position
            x = np.random.uniform(-2.0, 2.0)
            theta = np.random.uniform(-np.pi, np.pi)  # Full 360° range
            x_dot = np.random.uniform(-1.5, 1.5)
            theta_dot = np.random.uniform(-2.0, 2.0)
            self.state = np.array([x, x_dot, theta, theta_dot])
        else:
            # Normal reset - small perturbations for initial training
            self.state = np.random.uniform(low=-0.45, high=0.45, size=(4,))
        
        self.steps = 0
        self.edge_timer = 0
        return self.state
    
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        attributes = np.array([self.cart_mass, self.pole_mass, self.pole_length, 
                              self.gravity, self.damping_cart, self.damping_pole])
        force = float(action)

        # RK4 integration for accurate physics
        k1 = self.__derivatives(self.state, force, attributes)
        k2 = self.__derivatives(self.state + self.dt/2 * k1, force, attributes)
        k3 = self.__derivatives(self.state + self.dt/2 * k2, force, attributes)
        k4 = self.__derivatives(self.state + self.dt * k3, force, attributes)
        self.state = self.state + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        # ✅ WRAP THETA to [-π, π] - this is the key insight!
        # This ensures 2π, 4π, 6π all map to 0 (same physical state)
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
        
        self.steps += 1
        x, x_dot, theta, theta_dot = self.state

        # Track time at edge
        if abs(x) > 2.5:
            self.edge_timer += 1
        else:
            self.edge_timer = 0

        # Termination - RELAXED for full-range training
        terminated = bool(
            x < -3 or x > 3 or  # Cart out of bounds
            self.steps >= self.max_steps or  # Time limit
            self.edge_timer > 100  # Very lenient - allow exploration at edges
            # NO angle termination - we want to learn to recover from any angle!
        )
        
        # Safety clipping
        max_x = 3.0
        max_v = 5.0
        self.state[0] = np.clip(self.state[0], -max_x, max_x)
        self.state[1] = np.clip(self.state[1], -max_v, max_v)

        # REWARD FUNCTION - works for full 360° range
        desired_x = 0.0
        
        if not terminated:
            # Angle reward - cosine-based (naturally handles full range)
            # cos(0) = 1 (upright), cos(π) = -1 (upside down)
            upright_reward = np.cos(theta)  # 1.0 when upright, -1.0 when upside down
            
            # Position reward - quadratic to strongly prefer center
            position_penalty = 0.5 * (abs(x) / 3.0) ** 2
            
            # Velocity penalties - want slow, controlled movement
            angular_vel_penalty = 0.02 * abs(theta_dot)
            linear_vel_penalty = 0.05 * abs(x_dot)
            
            # Strong exponential wall penalty
            if abs(x) > 2.0:
                wall_distance = (abs(x) - 2.0) / 1.0  # 0 to 1
                wall_penalty = 3.0 * (wall_distance ** 2)
            else:
                wall_penalty = 0
            
            # Combine rewards
            # Scale upright_reward to be dominant
            reward = (2.0 * upright_reward  # -2 to +2 based on angle
                     - position_penalty      # 0 to -0.5
                     - angular_vel_penalty   # velocity penalties
                     - linear_vel_penalty 
                     - wall_penalty)         # 0 to -3
            
        else:
            # Moderate termination penalty (not too harsh since we want exploration)
            reward = -5.0
            
        info = {"x": x, "theta": theta}
        return self.state, reward, terminated, info
        
    def __derivatives(self, state, force, attributes):
        x, x_dot, theta, theta_dot = state
        cart_mass, pole_mass, pole_length, gravity, damping_cart, damping_pole = attributes
        
        # Standard inverted pendulum equations
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        total_mass = cart_mass + pole_mass
        
        # These equations naturally work for any angle (use sin and cos)
        theta_ddot = (gravity * np.sin(theta) + 
                     np.cos(theta) * ((-force - pole_mass * pole_length * theta_dot**2 * np.sin(theta)) / total_mass)) / \
                     (pole_length * ((4.0 / 3.0) - (pole_mass * np.cos(theta)**2) / total_mass))
        
        x_ddot = (force + pole_mass * pole_length * 
                 (theta_dot**2 * np.sin(theta) - theta_ddot * np.cos(theta))) / total_mass

        # Apply damping
        x_ddot -= damping_cart * x_dot
        theta_ddot -= damping_pole * theta_dot

        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])