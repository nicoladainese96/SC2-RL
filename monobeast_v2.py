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

from torchbeast.core import environment
from torchbeast.core import file_writer
from torchbeast.core import prof
from torchbeast.core import vtrace

# SC stuff
from pysc2.env import sc2_env
from SC_Utils.game_utils import IMPALA_ObsProcesser_v2, FullObsProcesser
from AC_modules.IMPALA import IMPALA_AC_v2
import absl 
import sys
import numpy as np

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
parser.add_argument('--action_names', '-a_n', type=str, nargs='*', help='List of strings containing action names to use.', 
                    default= ['no_op','move_camera', 'select_point', 'select_rect', 'select_idle_worker', 'select_army', 
                              'Attack_screen','Attack_minimap', 'Build_Barracks_screen', 'Build_CommandCenter_screen',
                              'Build_Refinery_screen', 'Build_SupplyDepot_screen','Harvest_Gather_SCV_screen', 
                              'Harvest_Return_SCV_quick', 'HoldPosition_quick', 'Move_screen', 'Move_minimap',
                              'Rally_Workers_screen', 'Rally_Workers_minimap','Train_Marine_quick', 'Train_SCV_quick'])
# Agent arguments
#parser.add_argument('--conv_channels', type=int, help='Number of convolutional channels for screen+minimap output', default=32)
#parser.add_argument('--player_features', type=int, help='Number of features for the player features output', default=16)
#parser.add_argument('--n_features', type=int, help='Number of features of the non-spatial features', default=256)
parser.add_argument("--mode", default="train",
                    choices=["train", "test"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="./logs/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=4, type=int, metavar="N",
                    help="Number of actors (default: 4).")
parser.add_argument("--total_steps", default=12000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=60, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--num_learner_threads", "--num_threads", default=1, type=int, # old default was 2
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.0005,
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--optim", default="RMSprop",
                    type=str, help="Optimizer. Choose between RMSprop and Adam.")
parser.add_argument("--learning_rate", default=0.0007,#0.00048,
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

Buffers = typing.Dict[str, typing.List[torch.Tensor]] 

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

def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)

def compute_policy_gradient_loss(log_prob, advantages):
    log_prob = log_prob.view_as(advantages)
    return - torch.sum(log_prob * advantages.detach())


def act(
    flags,
    game_params,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers,
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        sc_env = init_game(game_params['env'], flags.map_name, random_seed=seed)
        obs_processer = IMPALA_ObsProcesser_v2(env=sc_env, action_table=model.action_table, **game_params['obs_processer'])
        env = environment.Environment_v2(sc_env, obs_processer, seed)
        # initial rollout starts here
        env_output = env.initial() 
        new_res = model.spatial_processing_block.new_res
        agent_state = model.spatial_processing_block.conv_lstm._init_hidden(batch_size=1, 
                                                                            image_size=(new_res,new_res)
                                                                           )
        
        with torch.no_grad():
            agent_output, new_agent_state = model.actor_step(env_output, *agent_state[0]) 

        agent_state = agent_state[0] # _init_hidden yields [(h,c)], whereas actor step only (h,c)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end. 
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                if key not in ['sc_env_action']: # no need to save this key on buffers
                    buffers[key][index][0, ...] = agent_output[key]
            
            # lstm state in syncro with the environment / input to the agent 
            # that's why agent_state = new_agent_state gets executed afterwards
            initial_agent_state_buffers[index][0][...] = agent_state[0]
            initial_agent_state_buffers[index][1][...] = agent_state[1]
            
            
            # Do new rollout.
            for t in range(flags.unroll_length):
                timings.reset()

                env_output = env.step(agent_output["sc_env_action"])
                
                timings.time("step")
                
                # update state
                agent_state = new_agent_state 
            
                with torch.no_grad():
                    agent_output, new_agent_state = model.actor_step(env_output, *agent_state)
                
                timings.time("model")
                
                #env_output = env.step(agent_output["sc_env_action"])

                #timings.time("step")

                for key in env_output:
                    buffers[key][index][t+1, ...] = env_output[key] 
                for key in agent_output:
                    if key not in ['sc_env_action']: # no need to save this key on buffers
                        buffers[key][index][t+1, ...] = agent_output[key] 
                # env_output will be like
                # s_{0}, ..., s_{T}
                # act_mask_{0}, ..., act_mask_{T}
                # discount_{0}, ..., discount_{T}
                # r_{-1}, ..., r_{T-1}
                # agent_output will be like
                # a_0, ..., a_T with a_t ~ pi(.|s_t)
                # log_pi(a_0|s_0), ..., log_pi(a_T|s_T)
                # so the learner can use (s_i, act_mask_i) to predict log_pi_i
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
    initial_agent_state_buffers,
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
    initial_agent_state = [torch.stack([initial_agent_state_buffers[m][i][0] for m in indices], axis=0)
                      for i in range(2)]
    #print("initial_agent_state[0].shape: ", initial_agent_state[0].shape)
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = [t.to(device=flags.device, non_blocking=True) for t in initial_agent_state]
    timings.time("device")
    return batch, initial_agent_state

def learn(
    flags,
    actor_model, # single actor model with shared memory? Confirm that?
    model,
    batch,
    initial_agent_state,
    optimizer,
    scheduler,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        
        learner_outputs = model.learner_step(batch, initial_agent_state) 
        
        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline_trg"][-1] # V_learner(s_T)
        entropy = learner_outputs['entropy']
        
        #rearranged_batch = {}
        #rearranged_batch['done'] = batch['done'][:-1] # done_{0}, ..., done_{T-1}
        #rearranged_batch['done'] = batch['done'][1:]
        #rearranged_batch['bootstrap'] = batch['bootstrap'][1:]
        #rearranged_batch['reward'] = batch['reward'][1:] # reward_{0}, ..., reward_{T-1}
        #rearranged_batch['log_prob'] = batch['log_prob'][:-1] # log_prob_{0}, ..., log_prob_{T-1}
        
        # gets [log_prob_{0}, ..., log_prob_{T-1}] and [V_{0},...,V_{T-1}]
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items() if key != 'entropy'}

        rewards = batch['reward'][1:]
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = rewards

        #discounts = (~rearranged_batch["done"]).float() * flags.discounting # 0 if done, gamma otherwise

        vtrace_returns = vtrace.from_logits(
            behavior_action_log_probs=batch['log_prob'][:-1], # actor
            target_action_log_probs=learner_outputs["log_prob"], # learner
            not_done=(~batch['done'][1:]).float(),
            bootstrap=batch['bootstrap'][1:],
            gamma=flags.discounting,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            values_trg=learner_outputs["baseline_trg"],
            bootstrap_value=bootstrap_value, # coming from the learner too
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["log_prob"],
            vtrace_returns.pg_advantages,
        )
       
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )

        entropy_loss = flags.entropy_cost * entropy
        total_loss = pg_loss + baseline_loss + entropy_loss
        # not every time we get an episode return because the unroll length is shorter than the episode length, 
        # so not every time batch['done'] contains some True entries
        episode_returns = batch["episode_return"][batch["done"]] # still to check, might be okay
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
        if flags.optim == "RMSprop":
            scheduler.step()
        actor_model.load_state_dict(model.state_dict())
        return stats


def create_buffers(
    flags, 
    screen_shape,
    minimap_shape,
    player_shape, 
    num_actions, 
    max_num_spatial_args, 
    max_num_categorical_args
) -> Buffers:
    """`flags` must contain unroll_length and num_buffers"""
    T = flags.unroll_length
    # specs is a dict of dict which containt the keys 'size' and 'dtype'
    specs = dict(
        screen_layers=dict(size=(T+1, *screen_shape), dtype=torch.float32), 
        minimap_layers=dict(size=(T+1, *minimap_shape), dtype=torch.float32),
        player_state=dict(size=(T+1, player_shape), dtype=torch.float32), 
        screen_layers_trg=dict(size=(T+1, *screen_shape), dtype=torch.float32), 
        minimap_layers_trg=dict(size=(T+1, *minimap_shape), dtype=torch.float32),
        player_state_trg=dict(size=(T+1, player_shape), dtype=torch.float32), 
        last_action=dict(size=(T+1,), dtype=torch.int64),
        action_mask=dict(size=(T+1, num_actions), dtype=torch.bool), 
        reward=dict(size=(T+1,), dtype=torch.float32),
        done=dict(size=(T+1,), dtype=torch.bool),
        bootstrap=dict(size=(T+1,), dtype=torch.bool),
        episode_return=dict(size=(T+1,), dtype=torch.float32),
        episode_step=dict(size=(T+1,), dtype=torch.int32),
        log_prob=dict(size=(T+1,), dtype=torch.float32),
        main_action=dict(size=(T+1,), dtype=torch.int64), 
        categorical_indexes=dict(size=(T+1, max_num_categorical_args), dtype=torch.int64),
        spatial_indexes=dict(size=(T+1, max_num_spatial_args), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers

def train(flags, game_params):  # pylint: disable=too-many-branches, too-many-statements
    """
    1. Init actor model and create_buffers()
    2. Starts 'num_actors' act() functions
    3. Init learner model and optimizer, loads the former on the GPU
    4. Launches 'num_learner_threads' threads executing batch_and_learn()
    5. train finishes when all batch_and_learn threads finish, i.e. when steps >= flags.total_steps
    """
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )
    print("checkpointpath: ", checkpointpath)
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

    model = IMPALA_AC_v2(env=env, device='cpu', **game_params['HPs']) 
    screen_shape = (game_params['HPs']['screen_channels'], *model.screen_res)
    minimap_shape = (game_params['HPs']['minimap_channels'], *model.screen_res)
    player_shape = game_params['HPs']['in_player']
    num_actions = model.action_space
    buffers = create_buffers(flags, 
                             screen_shape, 
                             minimap_shape,
                             player_shape, 
                             num_actions,
                             model.max_num_spatial_args, 
                             model.max_num_categorical_args) 
    
    model.share_memory() # see if this works out of the box for my A2C

    # Add initial RNN state.
    initial_agent_state_buffers = []
    new_res = model.spatial_processing_block.new_res
    for _ in range(flags.num_buffers):
        state = model.spatial_processing_block.conv_lstm._init_hidden(batch_size=1, 
                                                                      image_size=(new_res, new_res)
                                                                  )
        
        state = state[0] # [(h,c)] -> (h,c)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)
        
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
                buffers,
                initial_agent_state_buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)

    # only model loaded into the GPU ?
    learner_model = IMPALA_AC_v2(env=env, device='cuda', **game_params['HPs']).to(device=flags.device) 
    
    if flags.optim == "Adam":
        optimizer = torch.optim.Adam(
            learner_model.parameters(),
            lr=flags.learning_rate
        )
    else:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha,
        )

    def lr_lambda(epoch):
        """
        Linear schedule from 1 to 0 used only for RMSprop. 
        To be adjusted multiplying or not by batch size B depending on how the steps are counted.
        epoch = number of optimizer steps
        total_steps = optimizer steps * time steps * batch size
                    or optimizer steps * time steps
        """
        return 1 - min(epoch * T, flags.total_steps) / flags.total_steps #epoch * T * B if using B steps

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
                initial_agent_state_buffers,
                timings,
            )
            stats = learn(
                flags, model, learner_model, batch, agent_state, optimizer, scheduler
            )
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                step += T #* B # just count the parallel steps 
    # end batch_and_learn
    
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
        if flags.optim == "Adam":
            torch.save(
                {
                    "model_state_dict": model.state_dict(), 
                    "optimizer_state_dict": optimizer.state_dict(),
                    "flags": vars(flags),
                },
                checkpointpath, # only one checkpoint at the time is saved
            )
        else:
            torch.save(
                {
                    "model_state_dict": model.state_dict(), 
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "flags": vars(flags),
                },
                checkpointpath, # only one checkpoint at the time is saved
            )    
    # end checkpoint
    
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

            sps = (step - start_step) / (timer() - start_time) # steps per second
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
# (you have to add it manually when needed)

def test(flags, game_params, num_episodes: int = 100):
    if flags.xpid is None:
        raise Exception("Specify a experiment id with --xpid. `latest` option not working.")
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
        )

    sc_env = init_game(game_params['env'], flags.map_name)
    model = IMPALA_AC_v2(env=sc_env, device='cpu', **game_params['HPs']) # let's use cpu as default for test
    obs_processer = IMPALA_ObsProcesser_v2(env=sc_env, action_table=model.action_table, **game_params['obs_processer'])
    env = environment.Environment_v2(sc_env, obs_processer)
    model.eval() # disable dropout
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"]) 

    observation = env.initial() # env.reset
    returns = []
    # Init agent LSTM hidden state
    new_res = model.spatial_processing_block.new_res
    agent_state = model.spatial_processing_block.conv_lstm._init_hidden(batch_size=1, 
                                                                            image_size=(new_res,new_res)
                                                                           )
    agent_state = agent_state[0] # _init_hidden yields [(h,c)], whereas actor step only (h,c)
    
    while len(returns) < num_episodes:
        with torch.no_grad():
            agent_outputs, agent_state = model.actor_step(observation, *agent_state) 
        observation = env.step(agent_outputs["sc_env_action"])
        if observation["done"].item():
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
    env.close()
    returns = np.array(returns)
    logging.info(
        "Average returns over %i episodes: %.2f (std %.2f) ", num_episodes, returns.mean(), returns.std()
    )
    print("Saving to file")
    np.save('%s/%s/test_results'%(flags.savedir, flags.xpid), returns)

def main(flags):
    assert flags.optim in ['RMSprop', 'Adam'], \
        "Expected --optim to be one of [RMSprop, Adam], got "+flags.optim
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

    HPs = dict(action_names=flags.action_names,
               screen_channels=screen_channels+1, # counting binary mask tiling
               minimap_channels=minimap_channels+1, # counting binary mask tiling
               encoding_channels=32,
               in_player=in_player
              )
    game_params['HPs'] = HPs
    
    if flags.mode == "train":
        train(flags, game_params)
    else:
        test(flags, game_params)


if __name__ == "__main__":
    start = time.time()
    flags, unknown_flags = parser.parse_known_args()  # Let argparse parse known flags from sys.argv.
    absl.flags.FLAGS(sys.argv[:1] + unknown_flags)  # Let absl.flags parse the rest.
    main(flags)
    elapsed_time = time.time() - start
    print("Elapsed time: %.2f min"%(elapsed_time/60) )


