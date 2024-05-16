import numpy as np
import turtle
import argparse
import time
from maze import Maze, Particle, Robot, WeightedDistribution, weight_gaussian_kernel
from kalman import *
from fuzzy_based import *





# Jaskaran Singh 20583594
# Environment and Part of the particle Filter Taken from Lei Mao, University of Chicago
# Fuzzy Controller, Kaplan Filter and rest of the code Built by Me




def particle_main(window_width, window_height, num_particles, sensor_limit_ratio, grid_height, grid_width, num_rows, num_cols,
         wall_prob, random_seed, robot_speed, kernel_sigma, particle_show_frequency):

    start_time = time.time()  # Start timing
    sensor_limit = sensor_limit_ratio * max(grid_height * num_rows, grid_width * num_cols)

    window = turtle.Screen()

    window.addshape('smallcar.gif')

    window.setup(width=window_width, height=window_height)

    world = Maze(grid_height=grid_height, grid_width=grid_width, num_rows=num_rows, num_cols=num_cols, wall_prob=wall_prob, random_seed=random_seed)

    x = np.random.uniform(0, world.width)
    y = np.random.uniform(0, world.height)
    bob = Robot(x=x, y=y, maze=world, speed=robot_speed, sensor_limit=sensor_limit)

    particles = list()
    for i in range(num_particles):
        x = np.random.uniform(0, world.width)
        y = np.random.uniform(0, world.height)
        particles.append(Particle(x=x, y=y, maze=world, sensor_limit=sensor_limit))
    time.sleep(1)
    world.show_maze()
    num_updates = 0

    while True:
        print(robot_speed)
        print(particle_show_frequency)
        estimated_position = calculate_estimated_position(particles)
        actual_position = (bob.x, bob.y)
        num_updates = num_updates + 1
        if is_successful(estimated_position, actual_position):
            elapsed_time = time.time() - start_time

            # Calculate Efficiency Score
            efficiency_score = num_particles / (elapsed_time + 1)

            # Calculate Computation Score
            computation_score = num_updates / (elapsed_time + 1)

            print(f"Location successfully identified in {elapsed_time:.2f} seconds.")
            print(f"Efficiency Score: {efficiency_score:.2f}%.")
            print(f"Computation Score: {computation_score:.2f}%.")
            return elapsed_time, efficiency_score, computation_score

        readings_robot = bob.read_sensor(maze=world)

        particle_weight_total = 0
        for particle in particles:
            readings_particle = particle.read_sensor(maze=world)
            particle.weight = weight_gaussian_kernel(x1=readings_robot, x2=readings_particle, std=kernel_sigma)
            particle_weight_total += particle.weight
        #
        world.show_particles(particles=particles, show_frequency=particle_show_frequency)
        world.show_robot(robot=bob)
        world.show_estimated_location(particles=particles)
        world.clear_objects()

        # Make sure normalization is not divided by zero
        if particle_weight_total == 0:
            particle_weight_total = 1e-8

        # Normalize particle weights
        for particle in particles:
            particle.weight /= particle_weight_total

        # Resampling particles
        distribution = WeightedDistribution(particles=particles)
        particles_new = list()

        for i in range(num_particles):

            particle = distribution.random_select()

            if particle is None:
                x = np.random.uniform(0, world.width)
                y = np.random.uniform(0, world.height)
                particles_new.append(Particle(x=x, y=y, maze=world, sensor_limit=sensor_limit))

            else:
                particles_new.append(Particle(x=particle.x, y=particle.y, maze=world, heading=particle.heading,
                                              sensor_limit=sensor_limit, noisy=True))

        particles = particles_new

        heading_old = bob.heading
        bob.move(maze=world)
        heading_new = bob.heading
        dh = heading_new - heading_old
        #
        for particle in particles:
            particle.heading = (particle.heading + dh) % 360
            particle.try_move(maze=world, speed=bob.speed)


def kalman_main(window_width, window_height, grid_height, grid_width, num_rows, num_cols, wall_prob, random_seed, robot_speed, kernel_sigma):
    window = turtle.Screen()
    window.addshape('smallcar.gif')

    window.setup(width=window_width, height=window_height)
    start_time = time.time()  # Start timing
    bob_turtle = turtle.Turtle()
    bob_turtle.speed(1)  # Set drawing speed to slow for visibility


    world = Maze(grid_height=grid_height, grid_width=grid_width, num_rows=num_rows, num_cols=num_cols, wall_prob=wall_prob, random_seed=random_seed)
    x = np.random.uniform(0, world.width)
    y = np.random.uniform(0, world.height)
    bob = Robot(x=x, y=y, maze=world, speed=robot_speed)

    # Define initial conditions for Kalman Filter
    initial_state = np.array([x, y, 0, 0])  # [x, y, dx, dy]
    initial_covariance = np.eye(4) * 1000  # Large initial uncertainty
    process_noise = np.eye(4)
    measurement_noise = np.eye(2) * 10
    measurement_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    kf = KalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise, measurement_matrix)

    num_matrix_updates = 1
    time.sleep(1)
    world.show_maze()

    while True:
        # Simulate robot movement
        bob.move(world)  # Pass the maze object here
        dx, dy = bob.speed * np.cos(bob.heading), bob.speed * np.sin(bob.heading)
        kf.predict(np.eye(4), np.array([dx, dy, 0, 0]))
        num_matrix_updates = num_matrix_updates +1
        bob_turtle.goto(bob.x, bob.y)  # Move turtle to bob's position
        bob_turtle.dot(4, 'red')  # Mark bob's position

        # Obtain measurement
        measurement = np.array([bob.x + np.random.randn(), bob.y + np.random.randn()])
        kf.update(measurement)

        # Display estimated position
        estimated_position = kf.state[:2]
        actual_position = np.array([bob.x, bob.y])
        print(f"Kalman Filter - Estimated Position: {estimated_position[0]:.2f}, {estimated_position[1]:.2f}")

        if is_successful(estimated_position, actual_position):
            elapsed_time = time.time() - start_time

            # Similar calculation for the Kalman filter
            computation_score = num_matrix_updates / (elapsed_time + 1)

            print(f"State successfully identified in {elapsed_time:.2f} seconds with an computation score of {computation_score:.2f}%.")
            return computation_score


        world.show_robot(robot=bob)
        world.clear_objects()


        estimated_x, estimated_y = kf.state[0], kf.state[1]
        kalman_turtle = turtle.Turtle()
        kalman_turtle.shape('circle')  # Set the shape to 'circle'
        kalman_turtle.color('blue')  # Set the outline color
        kalman_turtle.penup()
        kalman_turtle.goto(estimated_x, estimated_y)  # Position the turtle
        kalman_turtle.pendown()
        kalman_turtle.pensize(2)  # Set the thickness of the circle's outline
        kalman_turtle.circle(9)  # Draw a hollow circle with a specific radius
        kalman_turtle.penup()  # Lift the pen up after drawing
        kalman_turtle.hideturtle()  # Hide the turtle after drawing



        kalman_turtle.goto(estimated_x, estimated_y)
        # kalman_turtle.stamp()  # Use stamp to place a hollow circle at the location
        # kalman_turtle.dot(2)
        turtle.update()  # Make sure to update the screen
        time.sleep(0.33)  # Slow down the loop for visibility

    turtle.done()



def fuzzy_particle_main(window_width, window_height, num_particles, sensor_limit_ratio, grid_height, grid_width, num_rows, num_cols,
         wall_prob, random_seed, robot_speed, kernel_sigma, particle_show_frequency):
    start_time = time.time()  # Start timing
    sensor_limit = sensor_limit_ratio * max(grid_height * num_rows, grid_width * num_cols)

    window = turtle.Screen()

    window.addshape('smallcar.gif')

    window.setup(width=window_width, height=window_height)

    world = Maze(grid_height=grid_height, grid_width=grid_width, num_rows=num_rows, num_cols=num_cols, wall_prob=wall_prob, random_seed=random_seed)

    x = np.random.uniform(0, world.width)
    y = np.random.uniform(0, world.height)
    bob = FuzzyRobot(x=x, y=y, maze=world, speed=robot_speed, sensor_limit=sensor_limit)

    particles = list()
    for i in range(num_particles):
        x = np.random.uniform(0, world.width)
        y = np.random.uniform(0, world.height)
        particles.append(Particle(x=x, y=y, maze=world, sensor_limit=sensor_limit))

    time.sleep(1)
    world.show_maze()

    while True:
        print(robot_speed)
        print(particle_show_frequency)
        estimated_position = calculate_estimated_position(particles)
        actual_position = (bob.x, bob.y)
        if is_successful(estimated_position, actual_position):
            elapsed_time = time.time() - start_time
            print(f"Location successfully identified in {elapsed_time:.2f} seconds.")
            break

        readings_robot = bob.read_sensor(maze=world)

        particle_weight_total = 0
        for particle in particles:
            readings_particle = particle.read_sensor(maze=world)
            particle.weight = weight_gaussian_kernel(x1=readings_robot, x2=readings_particle, std=kernel_sigma)
            particle_weight_total += particle.weight
        #
        world.show_particles(particles=particles, show_frequency=particle_show_frequency)
        world.show_robot(robot=bob)
        world.show_estimated_location(particles=particles)
        world.clear_objects()

        # Make sure normalization is not divided by zero
        if particle_weight_total == 0:
            particle_weight_total = 1e-8

        # Normalize particle weights
        for particle in particles:
            particle.weight /= particle_weight_total

        # Resampling particles
        distribution = WeightedDistribution(particles=particles)
        particles_new = list()

        for i in range(num_particles):

            particle = distribution.random_select()

            if particle is None:
                x = np.random.uniform(0, world.width)
                y = np.random.uniform(0, world.height)
                particles_new.append(Particle(x=x, y=y, maze=world, sensor_limit=sensor_limit))

            else:
                particles_new.append(Particle(x=particle.x, y=particle.y, maze=world, heading=particle.heading,
                                              sensor_limit=sensor_limit, noisy=True))

        particles = particles_new

        heading_old = bob.heading
        bob.move(maze=world)
        heading_new = bob.heading
        dh = heading_new - heading_old
        #
        for particle in particles:
            particle.heading = (particle.heading + dh) % 360
            particle.try_move(maze=world, speed=bob.speed)






def calculate_estimated_position(particles):
    """ Calculate the weighted average position of the particles. """
    total_weight = sum(p.weight for p in particles)
    if total_weight == 0:
        return 0, 0  # Avoid division by zero
    x = sum(p.x * p.weight for p in particles) / total_weight
    y = sum(p.y * p.weight for p in particles) / total_weight
    return x, y


def is_successful(estimated, actual, threshold=5):
    """ Determine if the estimated position is within the threshold distance from the actual position. """
    distance = np.sqrt((estimated[0] - actual[0]) * 2 + (estimated[1] - actual[1]) * 2)
    print(distance)
    return distance <= threshold


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Particle filter in maze.')

    window_width_default = 800
    window_height_default = 800
    num_particles_default = 1000
    sensor_limit_ratio_default = 0.3
    grid_height_default = 100
    grid_width_default = 100
    num_rows_default = 25
    num_cols_default = 25
    wall_prob_default = 0.25
    random_seed_default = 100
    robot_speed_default = 10
    kernel_sigma_default = 500
    particle_show_frequency_default = 10

    parser.add_argument('--window_width', type=int, help='Window width.', default=window_width_default)
    parser.add_argument('--window_height', type=int, help='Window height.', default=window_height_default)
    parser.add_argument('--num_particles', type=int, help='Number of particles used in particle filter.',
                        default=num_particles_default)
    parser.add_argument('--sensor_limit_ratio', type=float,
                        help='Distance limit of sensors (real value: 0 - 1). 0: Useless sensor; 1: Perfect sensor.',
                        default=sensor_limit_ratio_default)
    parser.add_argument('--grid_height', type=int, help='Height for each grid of maze.', default=grid_height_default)
    parser.add_argument('--grid_width', type=int, help='Width for each grid of maze.', default=grid_width_default)
    parser.add_argument('--num_rows', type=int, help='Number of rows in maze', default=num_rows_default)
    parser.add_argument('--num_cols', type=int, help='Number of columns in maze', default=num_cols_default)
    parser.add_argument('--wall_prob', type=float, help='Wall probability of a random maze.', default=wall_prob_default)
    parser.add_argument('--random_seed', type=int, help='Random seed for random maze and particle filter.',
                        default=random_seed_default)
    parser.add_argument('--robot_speed', type=int, help='Robot movement speed in maze.', default=robot_speed_default)
    parser.add_argument('--kernel_sigma', type=int, help='Standard deviation for Gaussian distance kernel.',
                        default=kernel_sigma_default)
    parser.add_argument('--particle_show_frequency', type=int, help='Frequency of showing particles on maze.',
                        default=particle_show_frequency_default)

    argv = parser.parse_args()

    window_width = argv.window_width
    window_height = argv.window_height
    num_particles = argv.num_particles
    sensor_limit_ratio = argv.sensor_limit_ratio
    grid_height = argv.grid_height
    grid_width = argv.grid_width
    num_rows = argv.num_rows
    num_cols = argv.num_cols
    wall_prob = argv.wall_prob
    random_seed = argv.random_seed
    robot_speed = argv.robot_speed
    kernel_sigma = argv.kernel_sigma
    particle_show_frequency = argv.particle_show_frequency


    print(particle_main(window_width=window_width, window_height=window_height, num_particles=num_particles, sensor_limit_ratio=sensor_limit_ratio, grid_height=grid_height, grid_width=grid_width, num_rows=num_rows, num_cols=num_cols, wall_prob=wall_prob, random_seed=random_seed, robot_speed=robot_speed, kernel_sigma=kernel_sigma, particle_show_frequency=particle_show_frequency))
    kalman_main(window_width=window_width, window_height=window_height, grid_height=grid_height, grid_width=grid_width, num_cols=num_cols,num_rows=num_rows, wall_prob=wall_prob, random_seed=random_seed, robot_speed=robot_speed, kernel_sigma=kernel_sigma)
    fuzzy_particle_main(window_width=window_width, window_height=window_height, num_particles=num_particles, sensor_limit_ratio=sensor_limit_ratio, grid_height=grid_height, grid_width=grid_width, num_rows=num_rows, num_cols=num_cols, wall_prob=wall_prob, random_seed=random_seed, robot_speed=robot_speed, kernel_sigma=kernel_sigma, particle_show_frequency=particle_show_frequency)
