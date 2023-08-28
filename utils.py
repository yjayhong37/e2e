import tensorflow as tf
import os
import cv2
from absl import logging


from tf_agents.drivers import py_driver
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils

from eval import StrategyPyTFEagerPolicy


def get_sampling_spec(action_tensor_spec, min_actions, max_actions, act_norm_layer, uniform_boundary_buffer=0.05):
    """Gets bounds of action space from normalization info and adds buffer."""
    def generate_boundary_buffered_limits(spec, min_action, max_action):
        # Optionally add a small buffer of extra acting range.
        action_range = max_action - min_action
        min_action -= action_range * uniform_boundary_buffer
        max_action += action_range * uniform_boundary_buffer

        # Clip this range to the envs' min/max.
        # There's no point in sampling outside of the envs' min/max.
        min_action = tf.maximum(spec.minimum, min_action)
        max_action = tf.minimum(spec.maximum, max_action)

        return min_action, max_action

    action_limit_nest = tf.nest.map_structure(generate_boundary_buffered_limits, action_tensor_spec,
                                              min_actions, max_actions)

    # Map up to the spec to avoid iterating over the tuples.
    buffered_min_actions = nest_utils.map_structure_up_to(action_tensor_spec, lambda a: a[0], action_limit_nest)
    buffered_max_actions = nest_utils.map_structure_up_to(action_tensor_spec, lambda a: a[1], action_limit_nest)

    normalized_min_actions = act_norm_layer(buffered_min_actions)[0]
    normalized_max_actions = act_norm_layer(buffered_max_actions)[0]

    def bounded_like(spec, min_action, max_action):
        return tensor_spec.BoundedTensorSpec(spec.shape, spec.dtype, minimum=min_action, maximum=max_action,
                                             name=spec.name)

    action_sampling_spec = tf.nest.map_structure(bounded_like, action_tensor_spec, normalized_min_actions,
                                                 normalized_max_actions)
    return action_sampling_spec


# Video --------------

class OssMp4VideoRecorder:
    """Open-source Mp4VideoRecorder for creating mp4 videos frame by frame."""

    def __init__(self, filepath, frame_rate):
        self.filepath = filepath
        self.frame_rate = frame_rate
        self.vid_writer = None
        basedir = os.path.dirname(self.filepath)
        if not os.path.isdir(basedir):
            os.system("mkdir -p " + basedir)
        self.last_frame = None  # buffer so we don't write the last one.

    def init_vid_writer(self, width, height):
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.vid_writer = cv2.VideoWriter(self.filepath, self.fourcc,
                                          self.frame_rate, (width, height))

    def add_frame(self, frame):
        """Adds a frame to the video recorder.

    Args:
      frame: numpy array of shape [height, width, 3] representing the frame
        to add to the video.
    """
        # make even to avoid codec issues
        (h, w, _) = frame.shape
        if (h % 2 != 0) or (w % 2 != 0):
            if h % 2 != 0:
                h -= 1
            if w % 2 != 0:
                w -= 1
            frame = frame[:h, :w, :]

        if self.vid_writer is None:
            self.init_vid_writer(w, h)

        # :facepalm: why did opencv ever choose BGR?
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self.last_frame is not None:
            self.vid_writer.write(self.last_frame)
        self.last_frame = frame

    def end_video(self):
        """Closes the video recorder and writes the frame buffer to disk."""
        self.vid_writer.release()


class Mp4VideoWrapper:
    """Open-source environment wrapper that adds frames to an mp4 video."""

    def __init__(self, env, frame_rate, frame_interval, video_filepath):
        self._video_recorder = OssMp4VideoRecorder(video_filepath, frame_rate)
        self.batched = False
        self.env = env
        self.env.episode_steps = 0
        self._frame_interval = frame_interval

    def reset(self):
        time_step = self.env.reset()
        self.env.episode_steps = 0
        self._add_frame()
        return time_step

    def step(self, action):
        time_step = self.env.step(action)
        if not self.env.episode_steps % self._frame_interval:
            self._add_frame()
        self.env.episode_steps += 1
        return time_step

    def close(self):
        if self._video_recorder is None:
            raise ValueError("Already ended this video! I'm a one-time-use wrapper.")
        self._video_recorder.end_video()
        self._video_recorder = None

    def _add_frame(self):
        frame = self.env.render()
        self._video_recorder.add_frame(frame)


def make_video(agent, env, root_dir, step, strategy):
    """Create video from policy"""
    policy = StrategyPyTFEagerPolicy(agent.policy, strategy=strategy)
    video_path = os.path.join(root_dir, 'videos', 'ttl=7d', 'vid_%d.mp4' % step)
    control_frequency = 30

    video_env = Mp4VideoWrapper(env, control_frequency, frame_interval=1, video_filepath=video_path)
    driver = py_driver.PyDriver(video_env, policy, observers=[], max_episodes=1)
    time_step = video_env.reset()
    initial_policy_state = policy.get_initial_state(1)
    driver.run(time_step, initial_policy_state)
    video_env.close()
    logging.info('Wrote video for step %d to %s', step, video_path)
