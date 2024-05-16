from main import *
import pandas as pd

def run_experiment1():
    results = []

    window_width_default = 600
    window_height_default = 600
    num_particles_default = 1000
    sensor_limit_ratio_default = 0.5
    grid_height_default = 50
    grid_width_default = 50
    num_rows_default = 15
    num_cols_default = 15
    wall_prob_default = 0.05
    random_seed_default = 100
    robot_speed_default = 50
    kernel_sigma_default = 100
    particle_show_frequency_default = 5


    num_particles_values = [1000, 1500, 2000, 2500, 3000, 3500, 4500]

    for num_particles in num_particles_values:
        avg_elapsed_time = 0
        avg_efficiency_score = 0
        avg_computation_score = 0

        for _ in range(20):  # Run 3 sets
            # Call particle_main and receive scores
            random_seed_default = random_seed_default + 1
            elapsed_time, efficiency_score, computation_score =  particle_main(window_width=window_width_default, window_height=window_height_default, num_particles=num_particles, sensor_limit_ratio=sensor_limit_ratio_default, grid_height=grid_height_default, grid_width=grid_width_default, num_rows=num_rows_default, num_cols=num_cols_default, wall_prob=wall_prob_default, random_seed=random_seed_default, robot_speed=robot_speed_default, kernel_sigma=kernel_sigma_default, particle_show_frequency=particle_show_frequency_default)

            # Aggregate scores
            avg_elapsed_time += elapsed_time
            avg_efficiency_score += efficiency_score
            avg_computation_score += computation_score

        # Calculate averages
        avg_elapsed_time /= 20
        avg_efficiency_score /= 20
        avg_computation_score /= 20

        results.append({
            "num_particles": num_particles,
            "avg_elapsed_time": avg_elapsed_time,
            "avg_efficiency_score": avg_efficiency_score,
            "avg_computation_score": avg_computation_score
        })



def run_experiment2():
    results = []

    window_width_default = 600
    window_height_default = 600
    num_particles_default = 3000
    sensor_limit_ratio_default = 0.5
    grid_height_default = 50
    grid_width_default = 50
    num_rows_default =5
    num_cols_default = 5
    wall_prob_default = 0.2
    random_seed_default = 100
    robot_speed_default = 50
    kernel_sigma_default = 100
    particle_show_frequency_default = 5


    robot_speed_values = [5, 10, 15, 20, 25, 35, 40, 50]

    for robot_speed in robot_speed_values:
        avg_elapsed_time = 0
        avg_efficiency_score = 0
        avg_computation_score = 0

        for _ in range(3):  # Run 3 sets
            # Call particle_main and receive scores
            random_seed_default = random_seed_default + 1
            elapsed_time, efficiency_score, computation_score =  particle_main(window_width=window_width_default, window_height=window_height_default, num_particles=num_particles_default, sensor_limit_ratio=sensor_limit_ratio_default, grid_height=grid_height_default, grid_width=grid_width_default, num_rows=num_rows_default, num_cols=num_cols_default, wall_prob=wall_prob_default, random_seed=random_seed_default, robot_speed=robot_speed, kernel_sigma=kernel_sigma_default, particle_show_frequency=particle_show_frequency_default)

            # Aggregate scores
            avg_elapsed_time += elapsed_time
            avg_efficiency_score += efficiency_score
            avg_computation_score += computation_score

        # Calculate averages
        avg_elapsed_time /= 20
        avg_efficiency_score /= 20
        avg_computation_score /= 20

        results.append({
            "robot_speed": robot_speed,
            "avg_elapsed_time": avg_elapsed_time,
            "avg_efficiency_score": avg_efficiency_score,
            "avg_computation_score": avg_computation_score
        })


def run_experiment3():
    results = []

    window_width_default = 600
    window_height_default = 600
    num_particles_default = 3000
    sensor_limit_ratio_default = 0.5
    grid_height_default = 50
    grid_width_default = 50
    num_rows_default = 5
    num_cols_default = 5
    wall_prob_default = 0.2
    random_seed_default = 100
    robot_speed_default = 50
    kernel_sigma_default = 100
    particle_show_frequency_default = 5

    robot_speed_values = [5, 10, 15, 20, 25, 35, 40, 50]

    for robot_speed in robot_speed_values:
        avg_elapsed_time = 0
        avg_efficiency_score = 0
        avg_computation_score = 0

        for _ in range(3):  # Run 3 sets
            # Call particle_main and receive scores
            random_seed_default = random_seed_default + 1


            computation_score = kalman_main(window_width=window_width_default,
                                                                              window_height=window_height_default,
                                                                              grid_height=grid_height_default,
                                                                              grid_width=grid_width_default,
                                                                              num_rows=num_rows_default,
                                                                              num_cols=num_cols_default,
                                                                              wall_prob=wall_prob_default,
                                                                              random_seed=random_seed_default,
                                                                              robot_speed=robot_speed,
                                                                              kernel_sigma=kernel_sigma_default)

            # Aggregate scores
            avg_computation_score += computation_score

        # Calculate averages
        avg_elapsed_time /= 3
        avg_efficiency_score /= 3
        avg_computation_score /= 3

        results.append({
            "robot_speed": robot_speed,
            "avg_computation_score": avg_computation_score
        })

if __name__ == "__main__":
    # run_experiment1()
    # run_experiment2()
    run_experiment3()



