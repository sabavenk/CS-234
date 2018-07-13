from gym_torcs import TorcsEnv
import numpy as np
import tensorflow as tf
import keras
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU

import os.path
import tqdm

def playGame(train=0):    #1 means Train, 0 means simply Run
    load_from = "."

    save_to = os.path.join("data", "saved")
    save_thresh = 100000 # Save if total reward for the episode is more

    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 29  #of sensors input

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 100000

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    actor  = ActorNetwork (sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    ou = OU().function       #Ornstein-Uhlenbeck Process
    buff = ReplayBuffer(BUFFER_SIZE)

    env = TorcsEnv(vision=False, throttle=True,gear_change=False)

    def state(ob):
        return np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

    def load_weights(dir):
        print("Loading weights from ", dir)
        try:
            actor .model.load_weights       (os.path.join(dir, "actormodel.h5"))
            critic.model.load_weights       (os.path.join(dir, "criticmodel.h5"))
            actor .target_model.load_weights(os.path.join(dir, "actormodel.h5"))
            critic.target_model.load_weights(os.path.join(dir, "criticmodel.h5"))
            print("Weight load successfully")
        except:
            print("Cannot find the weight")

    def save_weights(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

        print("Saving weights in ", dir)
        actor .model.save_weights(os.path.join(dir, "actormodel.h5"), overwrite=True)
        critic.model.save_weights(os.path.join(dir, "criticmodel.h5"), overwrite=True)

        with open(os.path.join(dir, "actormodel.json"), "w") as outfile:
            json.dump(actor.model.to_json(), outfile)

        with open(os.path.join(dir, "criticmodel.json"), "w") as outfile:
            json.dump(critic.model.to_json(), outfile)

    load_weights(load_from)
    # Generate a Torcs environment

    print("TORCS Experiment Start.")
    np.random.seed(1337)

    done = False
    step = 0
    epsilon = 1

    for episode in range(episode_count):

        print("Episode : " + str(episode) + " Replay Buffer " + str(buff.count()))

        ob = env.reset()
        s_t = state(ob)

        total_reward = 0.

        progress = tqdm.trange(max_steps, disable = not train)
        for _ in progress:
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])

            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train * max(epsilon, 0) * ou(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train * max(epsilon, 0) * ou(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = train * max(epsilon, 0) * ou(a_t_original[0][2], -0.1 , 1.00, 0.05)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info = env.step(a_t[0])
            s_t1 = state(ob)

            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer

            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states     = np.asarray([e[0] for e in batch])
            actions    = np.asarray([e[1] for e in batch])
            rewards    = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones      = np.asarray([e[4] for e in batch])

            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            if (train):
                loss += critic.model.train_on_batch([states,actions], y_t)

                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)

                actor .update_target()
                critic.update_target()

            total_reward += r_t
            s_t = s_t1

            progress.set_description( "Episode %4i, TR %6.0f, loss %7.0f" % (episode, total_reward, loss))
            #print("Episode", i, "Step", step, "Action", [ "%.3f" % x for x in a_t[0]], "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break

        #print("Episode %i, TOTAL REWARD %.0f" % (episode, total_reward))

        if train and total_reward > save_thresh:
            save_weights(save_to + str(episode))
            save_thresh = min(1000000, 2*save_thresh)

    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame(0)
