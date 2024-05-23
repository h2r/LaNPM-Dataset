import tensorflow as tf
import tensorflow_datasets as tfds
import rlds
from PIL import Image
import numpy as np
from tf_agents.policies import py_tf_eager_policy
import tf_agents
from tf_agents.trajectories import time_step as ts
#from IPython import display
from collections import defaultdict
import matplotlib.pyplot as plt
import tensorflow_hub as hub

def as_gif(images):
  # Render the images as the gif:
  images[0].save('/tmp/temp.gif', save_all=True, append_images=images[1:], duration=1000, loop=0)
  gif_bytes = open('/tmp/temp.gif','rb').read()
  return gif_bytes
saved_model_path = '/users/sjulian2/data/sjulian2/rt_1_x_tf_trained_for_002272480_step'

tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
    model_path=saved_model_path,
    load_specs_from_pbtxt=True,
    use_tf_function=True)
print('loaded checkpoint')

# Create a dataset object to obtain episode from

builder = tfds.builder_from_directory(builder_dir='/users/sjulian2/data/sjulian2/jaco_play/0.1.0')
#builder = tfds.builder_from_directory(builder_dir='/users/sjulian2/data/sjulian2/bridge/0.1.0')
ds = builder.as_dataset(split='train[:1]')
print("build dataset jaco_play")

# Perform one step of inference using dummy input

# Obtain a dummy observation, where the features are all 0
observation = tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(tfa_policy.time_step_spec.observation))

# Construct a tf_agents time_step from the dummy observation
tfa_time_step = ts.transition(observation, reward=np.zeros((), dtype=np.float32))

# Initialize the state of the policy
policy_state = tfa_policy.get_initial_state(batch_size=1)

# fine tune on tfa_policy, move dataset builder above
tfa_policy.trainable = True
print("Number of layers in the base model: ", len(tfa_policy.layers))
num_freeze_layers = round(len(tfa_policy)*(2/3))
# Freeze all the layers before the `fine_tune_at` layer
for layer in tfa_policy.layers[num_freeze_layers]:
  layer.trainable = False

# Run inference using the policy
action = tfa_policy.action(tfa_time_step, policy_state)

# builder code used to be here

# Obtain the steps from one episode from the dataset
ds_iterator = iter(ds)
episode = next(ds_iterator)
steps = episode[rlds.STEPS]

images = []

for step in steps:

  im = Image.fromarray(np.array(step['observation']['image']))
  images.append(im)

print(f'{len(images)} images')

#display.Image(as_gif(images))

def resize(image):
  image = tf.image.resize_with_pad(image, target_width=320, target_height=256)
  image = tf.cast(image, tf.uint8)
  return image

def terminate_bool_to_act(terminate_episode: tf.Tensor) -> tf.Tensor:
  return tf.cond(
      terminate_episode == tf.constant(1.0),
      lambda: tf.constant([1, 0, 0], dtype=tf.int32),
      lambda: tf.constant([0, 1, 0], dtype=tf.int32),
  )

def rescale_action_with_bound(
    actions: tf.Tensor,
    low: float,
    high: float,
    safety_margin: float = 0,
    post_scaling_max: float = 1.0,
    post_scaling_min: float = -1.0,
) -> tf.Tensor:
  """Formula taken from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range."""
  resc_actions = (actions - low) / (high - low) * (
      post_scaling_max - post_scaling_min
  ) + post_scaling_min
  return tf.clip_by_value(
      resc_actions,
      post_scaling_min + safety_margin,
      post_scaling_max - safety_margin,
  )

def rescale_action(action):
  """Rescales action."""

  action['world_vector'] = rescale_action_with_bound(
      action['world_vector'],
      low=-0.05,
      high=0.05,
      safety_margin=0.01,
      post_scaling_max=1.75,
      post_scaling_min=-1.75,
  )
# commenting out rotation_delta because it's not in the jaco_play dataset 
  '''action['rotation_delta'] = rescale_action_with_bound(
      action['rotation_delta'],
      low=-0.25,
      high=0.25,
      safety_margin=0.01,
      post_scaling_max=1.4,
      post_scaling_min=-1.4,
  )'''

  return action

def to_model_action(from_step):
  """Convert dataset action to model action. This function is specific for the Bridge dataset."""
  model_action = {}
  model_action['world_vector'] = from_step['action']['world_vector']
  model_action['terminate_episode'] = terminate_bool_to_act(
      from_step['action']['terminate_episode']
  )
  model_action['rotation_delta'] = from_step['action']['rotation_delta'] ##Missing for jaco_play_code

  open_gripper = from_step['action']['open_gripper'] ##misgging  for jaco  play

  possible_values = tf.constant([True, False], dtype=tf.bool)
  eq = tf.equal(possible_values, open_gripper)

  assert_op = tf.Assert(tf.reduce_any(eq), [open_gripper])

with tf.control_dependencies([assert_op]):
    model_action['gripper_closedness_action'] = tf.cond(
        # for open_gripper in bridge dataset,
        # 0 is fully closed and 1 is fully open
        open_gripper,
        # for Fractal data,
        # gripper_closedness_action = -1 means opening the gripper and
        # gripper_closedness_action = 1 means closing the gripper.
        lambda: tf.constant([-1.0], dtype=tf.float32),
        lambda: tf.constant([1.0], dtype=tf.float32),
    )

  model_action = rescale_action(model_action)

  return model_action

def _normalize(value, mean, std):
  return (value - mean) / std

#def jaco_play_map_action(to_step: rlds.Step, from_step: rlds.Step): // previous method signature
def jaco_play_map_action(from_step: rlds.Step):
  model_action = {}
  model_action['world_vector'] = _normalize(
      from_step['action']['world_vector'],
      mean=tf.constant(
          [0.00096585, -0.00580069, -0.00395066], dtype=tf.float32
      ),
      std=tf.constant([0.12234575, 0.09676983, 0.11155209], dtype=tf.float32),
  )
  model_action['gripper_closedness_action'] = from_step['action'][
      'gripper_closedness_action'
  ]
  model_action['terminate_episode'] = from_step['action'][
      'terminate_episode'
  ]

  model_action = rescale_action(model_action)
  return model_action

steps = list(steps)

# Load language model and

embed = hub.load(
    'https://tfhub.dev/google/universal-sentence-encoder-large/5')

# embed the task string
#print("before NL")
#print(steps[0][rlds.OBSERVATION])
#print("between two commands")
print(steps[0][rlds.OBSERVATION].keys())

print("decode language_instruction")
print(steps[0][rlds.OBSERVATION]['natural_language_instruction'].numpy().decode())

#print("now show rlds.language_instruction")
#print(steps[0][rlds.language_instruction])
#print("now show keys language instruction")
#print(steps[0][rlds.language_instruction].keys())
episode_natural_language_instruction = steps[0][rlds.OBSERVATION]['natural_language_instruction'].numpy().decode()
print("after getting one NL command")

def normalize_task_name(task_name):

  replaced = task_name.replace('_', ' ').replace('1f', ' ').replace(
      '4f', ' ').replace('-', ' ').replace('50',
                                           ' ').replace('55',
                                                        ' ').replace('56', ' ')
  return replaced.lstrip(' ').rstrip(' ')


natural_language_embedding = embed([normalize_task_name(episode_natural_language_instruction)])[0]

# %%time

policy_state = tfa_policy.get_initial_state(batch_size=1)

gt_actions = []
predicted_actions = []
images = []

for step in steps:

  image = resize(step[rlds.OBSERVATION]['image'])

  images.append(image)
  observation['image'] = image

  tfa_time_step = ts.transition(observation, reward=np.zeros((), dtype=np.float32))

  policy_step = tfa_policy.action(tfa_time_step, policy_state)
  action = policy_step.action
  policy_state = policy_step.state

  predicted_actions.append(action)
  gt_actions.append(jaco_play_map_action(step))

print("after steps loop")

action_name_to_values_over_time = defaultdict(list)
predicted_action_name_to_values_over_time = defaultdict(list)
figure_layout = ['terminate_episode_0', 'terminate_episode_1',
        'terminate_episode_2', 'world_vector_0', 'world_vector_1',
        'world_vector_2', 'rotation_delta_0', 'rotation_delta_1',
        'rotation_delta_2', 'gripper_closedness_action_0']
## TO-DO: might have to change to exclude rotation_delta
## UPDATE: removed rotation_delta from action_order
action_order = ['terminate_episode', 'world_vector', 'gripper_closedness_action']

for i, action in enumerate(gt_actions):

  for action_name in action_order:

    for action_sub_dimension in range(action[action_name].shape[0]):

      # print(action_name, action_sub_dimension)
      title = f'{action_name}_{action_sub_dimension}'

      action_name_to_values_over_time[title].append(action[action_name][action_sub_dimension])
      predicted_action_name_to_values_over_time[title].append(predicted_actions[i][action_name][action_sub_dimension])

figure_layout = [
    ['image'] * len(figure_layout),
    figure_layout
]
print("after traversing actions")
plt.rcParams.update({'font.size': 12})

stacked = tf.concat(tf.unstack(images[::3], axis=0), 1)

fig, axs = plt.subplot_mosaic(figure_layout)
fig.set_size_inches([45, 10])

for i, (k, v) in enumerate(action_name_to_values_over_time.items()):

  axs[k].plot(v, label='ground truth')
  axs[k].plot(predicted_action_name_to_values_over_time[k], label='predicted action')
  axs[k].set_title(k)
  axs[k].set_xlabel('Time in one episode')

axs['image'].imshow(stacked.numpy())
axs['image'].set_xlabel('Time in one episode (subsampled)')

plt.legend()

print("finished :)")