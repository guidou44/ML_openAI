import retro
import numpy as np
import cv2
import pickle
import neat


def eval_genomes(genomes, passed_config):

    for genome_id, genome in genomes:
        ob = env.reset()
        ac = env.action_space.sample()
        in_x, in_y, in_colors = env.observation_space.shape
        in_x = int(in_x / 8)
        in_y = int(in_y / 8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, passed_config)
        current_max_fitness = 0  # output from the network, how successful the genome was (from reward)
        fitness_current = 0
        frame = 0  # frame counter in emulator
        counter = 0
        x_pos = 0
        x_pos_max = 0
        done = False
        # cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        while not done:
            env.render()
            frame += 1

            ob = cv2.resize(ob, (in_x, in_y))  # resize to lower resolution level to facilitate neat (grey scale level)
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)  # gray scale
            ob = np.reshape(ob, (in_x, in_y))

            # What the network sees
            # scaled_img = cv2.cvtColor(ob, cv2.COLOR_BG2RGB)
            # scaled_img = cv2.resize(scaled_img, (in_x, in_y))
            # cv2.imshow('main', scaled_img)
            # cv2.waitKey(1)

            # convert neural network input in flat line (1D) of pixels instead of 2D
            image_array = ob.flatten()

            nn_output = net.activate(image_array)
            # print(image_array, nn_output)
            ob, rew, done, info = env.step(nn_output)

            x_pos = info['x']
            x_pos_end = info['screen_x_end']

            if x_pos > x_pos_max:
                fitness_current += 1
                x_pos_max = x_pos

            if x_pos == x_pos_end and x_pos > 500:
                fitness_current += 100000
                done = True

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if counter == 250:
                done = True

            if done:
                print(genome_id, fitness_current)

            genome.fitness = fitness_current


env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                     neat.DefaultStagnation, 'config-feedforward')

population = neat.Population(config)

population.add_reporter(neat.StdOutReporter(True))
statistics = neat.StatisticsReporter()
population.add_reporter(statistics)
population.add_reporter(neat.Checkpointer(10))

winner = population.run(eval_genomes)

with open('winner_sonicAI.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
