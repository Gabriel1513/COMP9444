import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA =  0.99# discount factor
INITIAL_EPSILON = 0.6 # starting value of epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period
BATCH_SIZE = 256
REPLAY_SIZE = 300  # experience replay buffer size

# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph

hidden_nodes = 128

layer_one_out = tf.layers.dense(state_in, hidden_nodes, 
        activation = tf.nn.tanh, 
        name = "q_network_hidden_layer_1")
layer_two_out = tf.layers.dense(layer_one_out, hidden_nodes,
        activation = tf.nn.tanh,
        name = "q_network_hidden_layer_2")


# TODO: Network outputs
q_values = tf.layers.dense(layer_two_out, ACTION_DIM,
        activation = None, 
        name = "q_network_output_layer")
q_action = tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

# TODO: Loss/Optimizer Definition
L2_BETA = 0.01

l2 = L2_BETA * sum(tf.nn.l2_loss(tf_var)
        for tf_var in tf.trainable_variables() if not ("bias" in tf_var.name))

loss = tf.reduce_mean(tf.square(target_in - q_action)) + l2
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

train_loss_summary_op = tf.summary.scalar("TrainingLoss", loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

def update_replay_buffer(replay_buffer, state, action, reward, next_state, done,
                         action_dim):
    """
    Update the replay buffer with provided input in the form:
    (state, one_hot_action, reward, next_state, done)
    Hint: the minibatch passed to do_train_step is one entry (randomly sampled)
    from the replay_buffer
    """

    buf_el = (state, action, reward, next_state, done)
    # append to buffer
    replay_buffer.append(buf_el)
    # Ensure replay_buffer doesn't grow larger than REPLAY_SIZE
    if len(replay_buffer) > REPLAY_SIZE:
        replay_buffer.pop(0)
    return None

replay_buffer = []
batch_presentations_count = 0

# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= (epsilon - FINAL_EPSILON) / EPSILON_DECAY_STEPS
    ep_reward = 0

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))
        ep_reward += reward

        update_replay_buffer(replay_buffer, state, action, reward,
                                 next_state, done, ACTION_DIM)

        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        target = reward + GAMMA * np.max(nextstate_q_values)
        

        # Do one training step
        session.run([optimizer], feed_dict={
            target_in: [target],
            action_in: [action],
            state_in: [state]
        })
            

        # Update
        state = next_state

        if (len(replay_buffer)>BATCH_SIZE):
            minibatch = random.sample(replay_buffer,BATCH_SIZE)
            #print(minibatch)
            state_batch = [data[0] for data in minibatch]
            action_batch = [data[1] for data in minibatch]
            reward_batch = [data[2] for data in minibatch]
            next_state_batch = [data[3] for data in minibatch]

            target_batch = []
            Q_value_batch = q_values.eval(feed_dict={
                state_in: next_state_batch
            })

            for i in range(0, BATCH_SIZE):
                if minibatch[i][4]:
                    target_batch.append(reward_batch[i])
                else:
                    target_val = reward_batch[i] + GAMMA * np.max(Q_value_batch[i])
                    target_batch.append(target_val)

            session.run([optimizer], feed_dict={
                target_in: target_batch,
                state_in: state_batch,
                action_in: action_batch
            })

        if done:
            break

        


    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
