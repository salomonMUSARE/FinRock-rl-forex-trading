import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from finrock.data_feeder import PdDataFeeder
from finrock.trading_env import TradingEnv
from finrock.scalers import MinMaxScaler, ZScoreScaler
from finrock.reward import SimpleReward, AccountValueChangeReward
from finrock.metrics import DifferentActions, AccountValue, MaxDrawdown, SharpeRatio
from finrock.indicators import BolingerBands, RSI, PSAR, SMA, MACD
from rockrl.utils.misc import MeanAverage
from rockrl.utils.memory import MemoryManager
from rockrl.tensorflow import PPOAgent
from rockrl.utils.vectorizedEnv import VectorizedEnv

# Suppress oneDNN optimization messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress TensorFlow informational logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Optional: Set TensorFlow logger level
tf.get_logger().setLevel('ERROR')

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

def actor_model(input_shape, action_space):
    input = layers.Input(shape=input_shape, dtype=tf.float32)
    x = layers.Flatten()(input)
    x = layers.Dense(512, activation='elu')(x)
    x = layers.Dense(256, activation='elu')(x)
    x = layers.Dense(64, activation='elu')(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(action_space, activation='softmax')(x)
    return models.Model(inputs=input, outputs=output)

def critic_model(input_shape):
    input = layers.Input(shape=input_shape, dtype=tf.float32)
    x = layers.Flatten()(input)
    x = layers.Dense(512, activation='elu')(x)
    x = layers.Dense(256, activation='elu')(x)
    x = layers.Dense(64, activation='elu')(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation=None)(x)
    return models.Model(inputs=input, outputs=output)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Required for Windows

    df = pd.read_csv('Datasets/pricedata.csv')
    df = df[:-1000]

    pd_data_feeder = PdDataFeeder(
        df,
        indicators=[
            BolingerBands(data=df, period=20, std=2),
            RSI(data=df, period=14),
            PSAR(data=df),
            MACD(data=df),
            SMA(data=df, period=7),
        ]
    )

    num_envs = 10
    env = VectorizedEnv(
        env_object=TradingEnv,
        num_envs=num_envs,
        data_feeder=pd_data_feeder,
        output_transformer=ZScoreScaler(),
        initial_balance=1000.0,
        max_episode_steps=1000,
        window_size=50,
        reward_function=AccountValueChangeReward(),
        metrics=[
            DifferentActions(),
            AccountValue(),
            MaxDrawdown(),
            SharpeRatio(),
        ]
    )

    action_space = env.action_space
    input_shape = env.observation_space.shape

    agent = PPOAgent(
        actor=actor_model(input_shape, action_space),
        critic=critic_model(input_shape),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        batch_size=128,
        lamda=0.95,
        kl_coeff=0.5,
        c2=0.01,
        writer_comment='ppo_sinusoid_discrete',
    )

    # Ensure the logdir exists before saving configs or models
    logdir = './ppo_sinusoidd'
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Save configuration files to logdir
    pd_data_feeder.save_config(logdir)
    env.env.save_config(logdir)

    # Save initial model
    initial_mean_reward = 0  # Placeholder for initial mean reward
    initial_account_value = 1000  # Starting account value
    agent.save_models(f'{logdir}/initial_model_epoch_0_reward_{int(initial_mean_reward)}_account_{int(initial_account_value)}')
    print("Saving model since it's the initial model")

    memory = MemoryManager(num_envs=num_envs)
    meanAverage = MeanAverage(best_mean_score_episode=1000)
    
    # Initialize the environment and get initial states
    states, infos = env.reset()
    rewards = 0.0
    epoch_counter = 0  # Initialize an epoch counter for manual control

    # Training loop
    while True:
        action, prob = agent.act(states)
        next_states, reward, terminated, truncated, infos = env.step(action)
        memory.append(states, action, reward, prob, terminated, truncated, next_states, infos)
        states = next_states

        for index in memory.done_indices():
            env_memory = memory[index]
            history = agent.train(env_memory)
            mean_reward = meanAverage(np.sum(env_memory.rewards))

            # Increment epoch after each training cycle
            epoch_counter += 1
            agent.epoch = epoch_counter  # Update agent.epoch manually

            # Save model at every 100 epochs
            if agent.epoch % 100 == 0:
                account_value = env_memory.infos[-1]["metrics"]['account_value']
                agent.save_models(f'{logdir}/checkpoint_model_epoch_{agent.epoch}_reward_{int(mean_reward)}_account_{int(account_value)}')
                print(f"Saving model at epoch {agent.epoch} as a checkpoint with reward {int(mean_reward)} and account value {int(account_value)}")

            # Save the best model if it achieves a new high mean reward
            if meanAverage.is_best(agent.epoch):
                account_value = env_memory.infos[-1]["metrics"]['account_value']
                agent.save_models(f'{logdir}/best_model_epoch_{agent.epoch}_reward_{int(mean_reward)}_account_{int(account_value)}')
                print(f"Saving model at epoch {agent.epoch} since it achieved a new best mean reward {int(mean_reward)} and account value {int(account_value)}")

            # Check for KL divergence condition and adjust learning rate if needed
            if history['kl_div'] > 0.05 and agent.epoch > 1000:
                agent.reduce_learning_rate(0.995, verbose=False)

            info = env_memory.infos[-1]
            print(agent.epoch, np.sum(env_memory.rewards), mean_reward, info["metrics"]['account_value'], history['kl_div'])
            agent.log_to_writer(info['metrics'])
            states[index], infos[index] = env.reset(index=index)

        # Exit condition based on the updated epoch_counter
        if agent.epoch >= 10000:
            break

    env.close()
    exit()
