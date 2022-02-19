from src.games.snake.env import SnakeEnv
from tensorflow.keras.optimizers import Adam
import os
import time
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from src.games.snake.model import get_model
from src.lib.helper import ImageProcessor
import pathlib

# current executing directory
CED = pathlib.Path(__file__).parent.resolve()
WEIGHTS_FILEPATH = os.path.join(
    CED, 'weights', "snake_weights_2000000_1645155465.12758.h5f")

IMG_SHAPE = (100, 100)
WINDOW_LENGTH = 4
INPUT_SHAPE = (WINDOW_LENGTH, IMG_SHAPE[0], IMG_SHAPE[1])
TRAIN = False

env = SnakeEnv()
nb_actions = env.action_space.n

model = get_model(INPUT_SHAPE, nb_actions)
processor = ImageProcessor(IMG_SHAPE)
nb_steps_policy = 1_000_000
nb_steps_train = 2_000_000
memory = SequentialMemory(
    limit=100000,
    window_length=WINDOW_LENGTH
)
policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(),
    attr='eps',
    value_max=1.,
    value_min=.1,
    value_test=.05,
    nb_steps=nb_steps_policy
)
dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    policy=policy,
    memory=memory,
    processor=processor,
    nb_steps_warmup=50000,
    gamma=.99,
    target_model_update=1000,
    train_interval=4,
    delta_clip=1
)

dqn.compile(Adam(learning_rate=.00025), metrics=['mae'])

if TRAIN:
    weights_filename = os.path.join(
        CED,
        'weights',
        f'{env.name}_weights_{nb_steps_train}_{time.time()}.h5f'
    )
    dqn.fit(
        env,
        nb_steps=nb_steps_train,
        log_interval=100000,
        visualize=False
    )
    dqn.save_weights(weights_filename, overwrite=True)

else:
    model.load_weights(WEIGHTS_FILEPATH)
    dqn.test(env, nb_episodes=1, visualize=True)
