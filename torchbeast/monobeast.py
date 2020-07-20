# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback
import typing

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from torchbeast import atari_wrappers
from torchbeast.core import environment
from torchbeast.core import file_writer
from torchbeast.core import prof
from torchbeast.core import vtrace

# SC stuff
from pysc2.env import sc2_env
from SC_Utils.game_utils import IMPALA_ObsProcesser
from AC_modules.IMPALA import IMPALA_AC
import absl 

# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent for StarCraftII Learning Environment")

# Game arguments
parser.add_argument('--res', type=int, help='Screen and minimap resolution', default=32)
parser.add_argument('--map_name', type=str, help='Name of the minigame', default='MoveToBeacon')
parser.add_argument('--select_all_layers', type=bool, help='If True, selects all useful layers of screen and minimap', default=True)
parser.add_argument('--screen_names', type=str, nargs='*', help='List of strings containing screen layers names to use. \
                    Overridden by select_all_layers=True', 
                    default=['visibility_map', 'player_relative', 'selected', 'unit_density', 'unit_density_aa'])
parser.add_argument('--minimap_names', type=str, nargs='*', help='List of strings containing minimap layers names to use. \
                    Overridden by select_all_layers=True', 
                    default=['visibility_map', 'camera'])
# Agent arguments
parser.add_argument('--conv_channels', type=int, help='Number of convolutional channels for screen+minimap output', default=32)
parser.add_argument('--player_features', type=int, help='Number of features for the player features output', default=16)
parser.add_argument('--n_features', type=int, help='Number of features of the non-spatial features', default=256)
parser.add_argument("--mode", default="train",
                    choices=["train", "test"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="~/logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=4, type=int, metavar="N",
                    help="Number of actors (default: 4).")
parser.add_argument("--total_steps", default=100000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--num_learner_threads", "--num_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.0006,
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.00048,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")
# yapf: enable


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]] #??

### Maybe import them from another file ###

def gen_PID():
    ID = ''.join([random.choice(string.ascii_letters) for _ in range(4)])
    ID = ID.upper()
    return ID

def init_game(game_params, map_name='MoveToBeacon', step_multiplier=8, **kwargs):

    race = sc2_env.Race(1) # 1 = terran
    agent = sc2_env.Agent(race, "Testv0") # NamedTuple [race, agent_name]
    agent_interface_format = sc2_env.parse_agent_interface_format(**game_params) #AgentInterfaceFormat instance

    game_params = dict(map_name=map_name, 
                       players=[agent], # use a list even for single player
                       game_steps_per_episode = 0,
                       step_mul = step_multiplier,
                       agent_interface_format=[agent_interface_format] # use a list even for single player
                       )  
    env = sc2_env.SC2Env(**game_params, **kwargs)

    return env

###

def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)

def compute_policy_gradient_loss(log_prob, advantages):
    log_prob = log_prob.view_as(advantages)
    return torch.sum(log_prob * advantages.detach())


def act(
    flags,
    game_params,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        sc_env = init_game(game_params['env'], flags.map_name, random_seed=seed)
        obs_processer = IMPALA_ObsProcesser(action_table=model.action_table, **game_params['obs_processer'])
        env = environment.Environment(sc_env, obs_processer)
        env_output = env.initial() 
        agent_output = model.step(env_output)
        # agent_output: dict(
        # log_prob=log_prob, 
        # baseline=baseline, 
        # action=action, 
        # entropy=entropy, 
        # sc2_env_action=sc2_env_action,
        #categorical_args_indexes=categorical_args_indexes,
        #spatial_args_indexes=spatial_args_indexes)

        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end. (why?)
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                if key not in ['sc2_env_action']:
                    buffers[key][index][0, ...] = agent_output[key]

            # Do new rollout.
            for t in range(game_params['unroll_length']):
                timings.reset()

                with torch.no_grad():
                    agent_output = model(env_output)

                timings.time("model")

                env_output = env.step(agent_output["sc2_env_action"])

                timings.time("step")

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    if key not in ['sc2_env_action']:
                        buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time("write")
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    timings,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    timings.time("device")
    return batch

def learn(
    flags,
    actor_model, # single actor model with shared memory? Confirm that?
    model,
    batch,
    optimizer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        learner_outputs = model.learner_step(batch)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}

        rewards = batch["reward"]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        discounts = (~batch["done"]).float() * flags.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_action_log_probs=batch["log_prob"], # actor
            target_action_log_probs=learner_outputs["log_prob"], # learner
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value, # coming from the learner too
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["log_prob"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        # here I would like to use just the log_prob of the main actions for the entropy regularization
        entropy_loss = flags.entropy_cost * learner_outputs["entropy"]

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": torch.mean(episode_returns).item(),
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()
        # similar to A3C; update learner, copy back to actor; 
        # but this updates all actors at the same time or just some of them? 
        # How is the update received? Only at a certain stage or asynchronously?
        actor_model.load_state_dict(model.state_dict())
        return stats


def create_buffers(flags, obs_shape, player_shape, num_actions, max_num_spatial_args, max_num_categorical_args) -> Buffers:
    """`flags` must contain unroll_length and num_buffers"""
    T = flags.unroll_length
    specs = dict(
        spatial_state=dict(size=(T + 1, *obs_shape), dtype=torch.float32), 
        player_state=dict(size=(T + 1, *player_shape), dtype=torch.float32), 
        action_mask=dict(size=(T + 1, num_actions), dtype=torch.bool), 
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        log_prob=dict(size=(T + 1,), dtype=torch.float32),
        #baseline=dict(size=(T + 1,), dtype=torch.float32),
        main_action=dict(size=(T + 1,), dtype=torch.int64), 
        categorical_args_indexes=dict(size=(T + 1, max_num_categorical_args), dtype=torch.int64),
        spatial_args_indexes=dict(size=(T + 1, max_num_spatial_args), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train(flags, game_params):  # pylint: disable=too-many-branches, too-many-statements
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    if flags.num_buffers is None:  # Set sensible default for num_buffers. IMPORTANT!!
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    env = init_game(game_params['env'], flags.map_name) 

    model = IMPALA_AC(env=env, device='cpu', **game_params['HPs']) 
    observation_shape = (game_params['HPs']['spatial_dict']['in_channels'], *model.screen_res)
    player_shape = game_params['HPs']['spatial_dict']['in_player']
    num_actions = len(model.action_names)
    buffers = create_buffers(flags, 
                             observation_shape, 
                             player_shape, 
                             num_actions,
                             model.max_num_spatial_args, 
                             model.max_num_categorical_args) 
    
    model.share_memory() # see if this works out of the box for my A2C

    actor_processes = []
    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                game_params,
                i,
                free_queue,
                full_queue,
                model, # with share memory
                buffers
            ),
        )
        actor.start()
        actor_processes.append(actor)

    # only model loaded into the GPU ?
    learner_model = IMPALA_AC(env=env, device='cpu', **game_params['HPs']).to(device=flags.device) 

    # no more Adam
    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        # linear schedule from 1 to 0 
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            batch, agent_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                timings,
            )
            stats = learn(
                flags, model, learner_model, batch, optimizer, scheduler
            )
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                step += T * B

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(), 
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()

# this test is thought as a stand-alone, but I prefer go get a test_env and an agent already loaded if possible 
# also this time it seems that is always working with the forward pass without a batch dimension 
#(you have to add it manually when needed)
def test(flags, num_episodes: int = 10):
    if flags.xpid is None:
        checkpointpath = "./latest/model.tar"
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
        )

    gym_env = create_env(flags) # init env
    env = environment.Environment(gym_env) # wrap it
    model = Net(gym_env.observation_space.shape, gym_env.action_space.n) # init model
    model.eval() # disable dropout
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"]) 

    observation = env.initial() # env.reset
    returns = []

    while len(returns) < num_episodes:
        if flags.mode == "test_render": # remove this
            env.gym_env.render()
        agent_outputs = model(observation)
        policy_outputs, _ = agent_outputs
        observation = env.step(policy_outputs["action"])
        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
    env.close()
    logging.info(
        "Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns)
    )

### Just focus on the in/out format of the forward method ###
class AtariNet(nn.Module):
    def __init__(self, observation_shape, num_actions, use_lstm=False):
        super(AtariNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        # Feature extraction.
        self.conv1 = nn.Conv2d(
            in_channels=self.observation_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4,
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer.
        self.fc = nn.Linear(3136, 512)

        # FC output size + one-hot of last action + last reward.
        core_output_size = self.fc.out_features + num_actions + 1

        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        x = inputs["frame"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )


Net = AtariNet

# here take inspiration from run.py to define all HPs (e.g. not only flags)
# also it would be better to have training with testing inside it, as I usually do
def main(flags):
    # Environment parameters
    RESOLUTION = flags.res
    game_params = {}
    game_params['env'] = dict(feature_screen=RESOLUTION, feature_minimap=RESOLUTION, action_space="FEATURES") 
    game_names = ['MoveToBeacon','CollectMineralShards','DefeatRoaches','FindAndDefeatZerglings',
                  'DefeatZerglingsAndBanelings','CollectMineralsAndGas','BuildMarines']
    map_name = flags.map_name
    game_params['map_name'] = map_name
    if map_name not in game_names:
        raise Exception("map name "+map_name+" not recognized.")
    
    # Action and state space params
    if flags.select_all_layers:
        obs_proc_params = {'select_all':True}
    else:
        obs_proc_params = {'screen_names':flags.screen_names, 'minimap_names':flags.minimap_names}
    game_params['obs_processer'] = obs_proc_params
    op = FullObsProcesser(**obs_proc_params)
    screen_channels, minimap_channels, in_player = op.get_n_channels()
    in_channels = screen_channels + minimap_channels 
    
    # A2C params
    spatial_model = net.FullyConvPlayerAndSpatial
    nonspatial_model = net.FullyConvNonSpatial
    # Internal features, passed inside a dictionary
    conv_channels = flags.conv_channels #32
    player_features = flags.player_features #16
    # Exposed features, passed outside of a dictionary
    n_channels = conv_channels + player_features #48
    n_features = flags.n_features #256

    spatial_dict = {"in_channels":in_channels, 'in_player':in_player, 
                    'conv_channels':conv_channels, 'player_features':player_features}
    nonspatial_dict = {'resolution':RESOLUTION, 'kernel_size':3, 'stride':2, 'n_channels':n_channels}

    HPs = dict(spatial_model=spatial_model, nonspatial_model=nonspatial_model,
           n_features=n_features, n_channels=n_channels, 
           spatial_dict=spatial_dict, nonspatial_dict=nonspatial_dict)
    game_params['HPs'] = HPs
    
    if flags.mode == "train":
        train(flags, game_params)
    else:
        test(flags)


if __name__ == "__main__":
    flags, unknown_flags = parser.parse_known_args()  # Let argparse parse known flags from sys.argv.
    absl.flags.FLAGS(sys.argv[:1] + unknown_flags)  # Let absl.flags parse the rest.
    main(flags)
