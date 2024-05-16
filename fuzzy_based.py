
import numpy as np
import turtle
import bisect
import argparse
from skfuzzy import control as ctrl
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl





class FuzzyParticle(object):

    def __init__(self, x, y, maze, heading=None, weight=1.0, sensor_limit=None, noisy=False):

        if heading is None:
            heading = np.random.uniform(0, 360)

        self.x = x
        self.y = y
        self.heading = heading
        self.weight = weight
        self.maze = maze
        self.sensor_limit = sensor_limit

        if noisy:
            std = max(self.maze.grid_height, self.maze.grid_width) * 0.2
            self.x = self.add_noise(x=self.x, std=std)
            self.y = self.add_noise(x=self.y, std=std)
            self.heading = self.add_noise(x=self.heading, std=360 * 0.05)

        self.fix_invalid_particles()
        self.setup_fuzzy_control()
        self.last_proximity = None
        self.last_action = 0
        self.move_count = 0

    def setup_fuzzy_control(self):
        # Proximity and action definitions
        proximity = ctrl.Antecedent(np.arange(0, 100, 1), 'proximity')
        action = ctrl.Consequent(np.arange(-180, 181, 1), 'action')

        # Define the membership functions more sharply
        proximity['close'] = fuzz.trimf(proximity.universe, [0, 0, 10])
        proximity['medium'] = fuzz.trimf(proximity.universe, [5, 20, 35])
        proximity['far'] = fuzz.trimf(proximity.universe, [30, 100, 100])

        action['sharp_left'] = fuzz.trimf(action.universe, [-180, -180, -90])
        action['left'] = fuzz.trimf(action.universe, [-100, -50, -10])
        action['forward'] = fuzz.trimf(action.universe, [-5, 0, 5])
        action['right'] = fuzz.trimf(action.universe, [10, 50, 100])
        action['sharp_right'] = fuzz.trimf(action.universe, [90, 180, 180])

        # Fuzzy rules to strongly favor forward movement
        rule1 = ctrl.Rule(proximity['close'], action['sharp_right'])  # Sharp right if very close to an obstacle
        rule2 = ctrl.Rule(proximity['medium'], action['forward'])  # Forward if medium close to an obstacle
        rule3 = ctrl.Rule(proximity['far'], action['forward'])  # Always move forward if far from an obstacle

        self.moving = ctrl.ControlSystem([rule1, rule2, rule3])
        self.moving_sim = ctrl.ControlSystemSimulation(self.moving)


    def fix_invalid_particles(self):

        # Fix invalid particles
        if self.x < 0:
            self.x = 0
        if self.x > self.maze.width:
            self.x = self.maze.width * 0.9999
        if self.y < 0:
            self.y = 0
        if self.y > self.maze.height:
            self.y = self.maze.height * 0.9999
        '''
        if self.heading > 360:
            self.heading -= 360
        if self.heading < 0:
            self.heading += 360
        '''
        self.heading = self.heading % 360

    @property
    def state(self):

        return (self.x, self.y, self.heading)

    def add_noise(self, x, std):

        return x + np.random.normal(0, std)

    def read_sensor(self, maze):
        # Assuming maze.distance_to_walls returns an array of distances in all directions
        distances = maze.distance_to_walls(coordinates=(self.x, self.y))
        # Select minimum distance for simplicity in this example
        proximity = min(distances)  # Ensure this is a single scalar value
        return proximity

    def try_move(self, speed, maze):
        # Read sensors to get proximity
        proximity = self.read_sensor_prox(maze)

        # Feed into fuzzy control system
        self.moving_sim.input['proximity'] = proximity
        self.moving_sim.compute()

        # Get new heading based on fuzzy output
        turn_angle = self.moving_sim.output['action']
        new_heading = (self.heading + turn_angle) % 360
        rad = np.radians(new_heading)

        dx = np.cos(rad) * speed
        dy = np.sin(rad) * speed

        new_x = self.x + dx
        new_y = self.y + dy

        # Check if the new position is within the grid and not blocked
        gj1 = int(self.x // maze.grid_width)
        gi1 = int(self.y // maze.grid_height)
        gj2 = int(new_x // maze.grid_width)
        gi2 = int(new_y // maze.grid_height)

        if gj2 < 0 or gj2 >= maze.num_cols or gi2 < 0 or gi2 >= maze.num_rows:
            return False  # Out of bounds

        # Check grid movement validity
        if gi1 == gi2 and gj1 == gj2:
            self.x, self.y, self.heading = new_x, new_y, new_heading
            return True

        # Check if moving to a new grid cell is allowed
        if maze.maze[gi2, gj2] == 0:  # Assuming 0 means wall
            return False

        self.x, self.y, self.heading = new_x, new_y, new_heading
        return True

class FuzzyRobot(FuzzyParticle):

    def __init__(self, x, y, maze, heading=None, speed=1.0, sensor_limit=None, noisy=True):

        super(FuzzyRobot, self).__init__(x=x, y=y, maze=maze, heading=heading, sensor_limit=sensor_limit, noisy=noisy)
        self.step_count = 0
        self.noisy = noisy
        self.time_step = 0
        self.speed = speed

    def choose_random_direction(self):

        self.heading = np.random.uniform(0, 360)

    def add_sensor_noise(self, x, z=0.05):
        # Calculate standard deviation as a fraction of x
        std = x * z / 2
        # Add Gaussian noise to x
        noisy_reading = x + np.random.normal(0, std)
        return noisy_reading

    def read_sensor_prox(self, maze):
        # Get the distance to the closest wall
        proximity = min(maze.distance_to_walls(coordinates=(self.x, self.y)))

        # Apply sensor noise if required
        if self.noisy:
            proximity = self.add_sensor_noise(proximity)

        print("***")
        print(proximity )
        print("***")
        return proximity
    def read_sensor(self, maze):

        readings = maze.distance_to_walls(coordinates=(self.x, self.y))

        heading = self.heading % 360

        # Remove the compass from particle
        if heading >= 45 and heading < 135:
            readings = readings
        elif heading >= 135 and heading < 225:
            readings = readings[-1:] + readings[:-1]
            # readings = [readings[3], readings[0], readings[1], readings[2]]
        elif heading >= 225 and heading < 315:
            readings = readings[-2:] + readings[:-2]
            # readings = [readings[2], readings[3], readings[0], readings[1]]
        else:
            readings = readings[-3:] + readings[:-3]
            # readings = [readings[1], readings[2], readings[3], readings[0]]

        if self.sensor_limit is not None:
            for i in range(len(readings)):
                if readings[i] > self.sensor_limit:
                    readings[i] = self.sensor_limit

        return readings

    def move(self, maze):

        while True:
            self.time_step += 1
            if self.try_move(speed=self.speed, maze=maze):
                break
            self.choose_random_direction()

class WeightedDistribution(object):

    def __init__(self, particles):

        accum = 0.0
        self.particles = particles
        self.distribution = list()
        for particle in self.particles:
            accum += particle.weight
            self.distribution.append(accum)

    def random_select(self):

        try:
            return self.particles[bisect.bisect_left(self.distribution, np.random.uniform(0, 1))]
        except IndexError:
            # When all particles have weights zero
            return None

def euclidean_distance(x1, x2):
    return np.linalg.norm(np.asarray(x1) - np.asarray(x2))


def weight_gaussian_kernel(x1, x2, std=10):
    distance = euclidean_distance(x1=x1, x2=x2)
    return np.exp(-distance ** 2 / (2 * std))
