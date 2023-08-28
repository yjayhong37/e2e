import random
import functools
import particle_3d
import os
import numpy as np
from absl import app, flags, logging
from tf_agents.policies import py_policy
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.metrics import py_metrics
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.trajectories import policy_step
from tf_agents.utils import example_encoding_dataset


FLAGS = flags.FLAGS
flags.DEFINE_boolean('make_video', False, 'Skips data generation & produces a video of the oracle instead')
flags.DEFINE_integer('num_videos', 50, 'Number of videos which are generated, requires --make_video')
flags.DEFINE_integer('num_jobs', 10, 'Number of data generation processes to run in parallel')
flags.DEFINE_integer('num_episodes', 200, 'Number of episodes to generate per job')


class ParticleOracle3D(py_policy.PyPolicy):
    """Creates Policy to generate training data"""
    def __init__(self, env, goal_threshold=0.01):
        super(ParticleOracle3D, self).__init__(env.time_step_spec(), env.action_spec())
        self._env = env
        self._np_random_state = np.random.RandomState(0)

        self.goal_threshold = goal_threshold
        self.reset()

    def reset(self):
        pass

    def _action(self, time_step, policy_state):
        if time_step.is_first():
            self.reset()

        obs = time_step.observation

        act = obs['pos_goal'] - obs['pos_agent']

        return policy_step.PolicyStep(action=act)


def create_episodes(dataset_path, num_episodes):
    """Create training data"""
    env = suite_gym.load('Particle3D-v1')
    policy = ParticleOracle3D(env)

    metrics = [py_metrics.AverageReturnMetric(buffer_size=num_episodes),
               py_metrics.AverageEpisodeLengthMetric(buffer_size=num_episodes)]
    env_metrics, success_metric = env.get_metrics(num_episodes)
    metrics += env_metrics

    observers = metrics[:]

    observers.append(example_encoding_dataset.TFRecordObserver(dataset_path, policy.collect_data_spec, py_mode=True,
                                                               compress_image=True))
    driver = py_driver.PyDriver(env, policy, observers, max_episodes=num_episodes)
    time_step = env.reset()
    initial_policy_state = policy.get_initial_state(1)

    driver.run(time_step, initial_policy_state)

    env.close()


def create_video():
    """Create video of the policy defined by the oracle class."""
    from utils import Mp4VideoWrapper

    np.random.seed(1)
    seeds = np.random.randint(size=FLAGS.num_videos, low=0, high=1000)

    for seed in seeds:
        env = suite_gym.load('Particle3D-v1')
        env.seed(seed)

        particle_policy = ParticleOracle3D(env)

        video_path = os.path.join('data', 'videos_3d', 'ttl=7d', 'vid_%d.mp4' % seed)
        control_frequency = 30

        video_env = Mp4VideoWrapper(env, control_frequency, frame_interval=1, video_filepath=video_path)
        driver = py_driver.PyDriver(video_env, particle_policy, observers=[], max_episodes=1)

        time_step = video_env.reset()

        initial_policy_state = particle_policy.get_initial_state(1)
        driver.run(time_step, initial_policy_state)
        video_env.close()
        logging.info('Wrote video for seed %d to %s', seed, video_path)

# Check and create the directory if it doesn't exist
def ensure_directory_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)



def main(_):
    if FLAGS.make_video:
        logging.info('Generating videos')
        create_video()
    else:
        dataset_path = 'data/3d_oracle_particle.tfrecord'
        ensure_directory_exists(dataset_path)
        dataset_split_path = os.path.splitext(dataset_path)

        context = multiprocessing.get_context()
        jobs = []
        for i in range(FLAGS.num_jobs):
            dataset_path = dataset_split_path[0] + '_%d' % i + dataset_split_path[1]
            ensure_directory_exists(dataset_path_i)
            job = context.Process(target=create_episodes, kwargs={"dataset_path": dataset_path,
                                                                  "num_episodes": FLAGS.num_episodes})
            job.start()
            jobs.append(job)

        for job in jobs:
            job.join()


if __name__ == "__main__":
    multiprocessing.handle_main(functools.partial(app.run, main))
