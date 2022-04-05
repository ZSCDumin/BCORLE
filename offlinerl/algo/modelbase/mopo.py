# MOPO: Model-based Offline Policy Optimization
# https://arxiv.org/abs/2005.13239
# https://github.com/tianheyu927/mopo

import torch
import numpy as np
from copy import deepcopy
from loguru import logger
from torch.functional import F
import copy

from tianshou.data import Batch

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.data import to_torch, sample
from offlinerl.utils.net.common import MLP, Net, Swish
from offlinerl.utils.net.tanhpolicy import TanhGaussianPolicy
from offlinerl.utils.exp import setup_seed

def algo_init(args):
    logger.info('Run algo_init function')

    setup_seed(args['seed'])
    
    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
    elif "task" in args.keys():
        #from offlinerl.utils.env import get_env_shape
        obs_shape, action_shape = 45, 1
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
    else:
        raise NotImplementedError
    
    transition = EnsembleTransition(obs_shape, action_shape, args['hidden_layer_size'], args['transition_layers'], args['transition_init_num']).to(args['device'])
    transition_optim = torch.optim.Adam(transition.parameters(), lr=args['transition_lr'], weight_decay=0.000075)

    net_a = Net(layer_num=args['hidden_layers'], 
                state_shape=obs_shape, 
                hidden_layer_size=args['hidden_layer_size'])

    actor = TanhGaussianPolicy(preprocess_net=net_a,
                               action_shape=action_shape,
                               hidden_layer_size=args['hidden_layer_size'],
                               conditioned_sigma=True).to(args['device'])

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args['actor_lr'])

    log_alpha = torch.zeros(1, requires_grad=True, device=args['device'])
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args["actor_lr"])

    q1 = MLP(obs_shape + action_shape, 1, args['hidden_layer_size'], args['hidden_layers'], norm=None, hidden_activation='swish').to(args['device'])
    q2 = MLP(obs_shape + action_shape, 1, args['hidden_layer_size'], args['hidden_layers'], norm=None, hidden_activation='swish').to(args['device'])
    critic_optim = torch.optim.Adam([*q1.parameters(), *q2.parameters()], lr=args['actor_lr'])

    return {
        "transition" : {"net" : transition, "opt" : transition_optim},
        "actor" : {"net" : actor, "opt" : actor_optim},
        "log_alpha" : {"net" : log_alpha, "opt" : alpha_optimizer},
        "critic" : {"net" : [q1, q2], "opt" : critic_optim},
    }

def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

class EnsembleLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=7):
        super().__init__()

        self.ensemble_size = ensemble_size

        self.register_parameter('weight', torch.nn.Parameter(torch.zeros(ensemble_size, in_features, out_features)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(ensemble_size, 1, out_features)))

        #torch.nn.init.trunc_normal_(self.weight, std=1/(2*in_features**0.5))

        self.select = list(range(0, self.ensemble_size))

    def forward(self, x):
        with torch.no_grad():
            weight = self.weight[self.select]
            bias = self.bias[self.select]

            if len(x.shape) == 2:
                x = torch.einsum('ij,bjk->bik', x, weight)
            else:
                x = torch.einsum('bij,bjk->bik', x, weight)

            x = x + bias

            return x

    def set_select(self, indexes):
        assert len(indexes) <= self.ensemble_size and max(indexes) < self.ensemble_size
        self.select = indexes

class EnsembleTransition(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_features, hidden_layers, ensemble_size=7, mode='local', with_reward=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.mode = mode
        self.with_reward = with_reward
        self.ensemble_size = ensemble_size

        self.activation = Swish()

        module_list = []
        for i in range(hidden_layers):
            if i == 0:
                module_list.append(EnsembleLinear(obs_dim + action_dim, hidden_features, ensemble_size))
            else:
                module_list.append(EnsembleLinear(hidden_features, hidden_features, ensemble_size))
        self.backbones = torch.nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(hidden_features, 2 * (obs_dim + self.with_reward), ensemble_size)

        self.register_parameter('max_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * 1, requires_grad=True))
        self.register_parameter('min_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * -5, requires_grad=True))

    def forward(self, obs_action):
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mu, logstd = torch.chunk(self.output_layer(output), 2, dim=-1)
        logstd = soft_clamp(logstd, self.min_logstd, self.max_logstd)
        if self.mode == 'local':
            if self.with_reward:
                obs, reward = torch.split(mu, [self.obs_dim, 1], dim=-1)
                obs = obs + obs_action[..., :self.obs_dim]
                mu = torch.cat([obs, reward], dim=-1)
            else:
                mu = mu + obs_action[..., :self.obs_dim]
        return torch.distributions.Normal(mu, torch.exp(logstd))

    def set_select(self, indexes):
        for layer in self.backbones:
            layer.set_select(indexes)
        self.output_layer.set_select(indexes)

class MOPOBuffer:
    def __init__(self, buffer_size):
        self.data = None
        self.buffer_size = int(buffer_size)

    def put(self, batch_data):
        batch_data.to_torch(device='cpu')

        if self.data is None:
            self.data = batch_data
        else:
            self.data.cat_(batch_data)
        
        if len(self) > self.buffer_size:
            self.data = self.data[len(self) - self.buffer_size : ]

    def __len__(self):
        if self.data is None: return 0
        return self.data.shape[0]

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self), size=(batch_size))
        return self.data[indexes]


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.transition = algo_init['transition']['net']
        self.transition_optim = algo_init['transition']['opt']
        self.selected_transitions = None

        self.actor = algo_init['actor']['net']
        self.actor_optim = algo_init['actor']['opt']

        self.log_alpha = algo_init['log_alpha']['net']
        self.log_alpha_optim = algo_init['log_alpha']['opt']

        self.q1, self.q2 = algo_init['critic']['net']
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.critic_optim = algo_init['critic']['opt']

        self.device = args['device']
        
    def pre_train(self, train_buffer):
        transition2 = self.train_transition(train_buffer)
        transition2.requires_grad_(False)
        return transition2

    def train(self, train_buffer,transition2):
        policy = self.train_policy(train_buffer, None, transition2, None)
    
    def get_policy(self):
        return self.actor

    def train_transition(self, buffer):
        data_size = len(buffer.replay["user_id"])
        val_size = min(int(data_size * 0.2) + 1, 1000)
        train_size = data_size - val_size
        train_splits, val_splits = torch.utils.data.random_split(range(data_size), (train_size, val_size))
        train_buffer = dict(state=None,action=None,reward=None,done=None,next_state=None)
        train_buffer['state'] = np.concatenate((buffer.replay["state"][train_splits.indices],buffer.replay["Lambda"][train_splits.indices]),1)
        train_buffer['action'] = buffer.replay["action"][train_splits.indices]
        train_buffer['reward'] = buffer.replay["reward"][train_splits.indices]
        train_buffer['done'] = buffer.replay["done"][train_splits.indices]
        train_buffer['next_state'] = np.concatenate((buffer.replay["next_state"][train_splits.indices],buffer.replay["Lambda"][train_splits.indices]),1)
        #train_buffer = buffer.new_batch_data[train_splits.indices]
        valdata = dict(state=None, action=None, reward=None, done=None, next_state=None)
        valdata['state'] =  np.concatenate((buffer.replay["state"][val_splits.indices],buffer.replay["Lambda"][val_splits.indices]),1)
        valdata['action'] = buffer.replay["action"][val_splits.indices]
        valdata['reward'] = buffer.replay["reward"][val_splits.indices]
        valdata['done'] = buffer.replay["done"][val_splits.indices]
        valdata['next_state'] = np.concatenate((buffer.replay["next_state"][val_splits.indices],buffer.replay["Lambda"][val_splits.indices]),1)
        #valdata = buffer.new_batch_data[val_splits.indices]
        batch_size = self.args['transition_batch_size']

        val_losses = [float('inf') for i in range(self.transition.ensemble_size)]

        epoch = 0
        cnt = 0
        while True:
            idxs = np.random.randint(len(train_buffer['state']), size=[self.transition.ensemble_size, len(train_buffer['state'])])
            for i in range(1):
                for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                    batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                    begin = batch_num * batch_size
                    if (batch_num+1) * batch_size > idxs.shape[-1]:
                        end = idxs.shape[-1]
                    else:
                        end = (batch_num+1) * batch_size
                    batch = dict(state=None, action=None, reward=None, done=None, next_state=None)
                    batch['state'] = np.concatenate((buffer.replay["state"][begin:end],buffer.replay["Lambda"][begin:end]),1)
                    batch['action'] = buffer.replay["action"][begin:end]
                    batch['reward'] = buffer.replay["reward"][begin:end]
                    batch['done'] = buffer.replay["done"][begin:end]
                    batch['next_state'] = np.concatenate((buffer.replay["next_state"][begin:end],buffer.replay["Lambda"][begin:end]),1)
                    #batch = train_buffer[batch_idxs]
                    self._train_transition(self.transition, batch, self.transition_optim)
                print(i)
            new_val_losses = self._eval_transition(self.transition, valdata)
            print(new_val_losses)

            change = False
            for i, new_loss, old_loss in zip(range(len(val_losses)), new_val_losses, val_losses):
                if new_loss < old_loss:
                    change = True
                    val_losses[i] = new_loss

            if change:
                cnt = 0
            else:
                cnt += 1

            if cnt >= 5:
                break
        
        val_losses = self._eval_transition(self.transition, valdata)
        indexes = self._select_best_indexes(val_losses, n=self.args['transition_select_num'])
        self.transition.set_select(indexes)
        return self.transition

    def train_policy(self, train_buffer, val_buffer, transition, callback_fn):
        real_batch_size = int(self.args['policy_batch_size'] * self.args['real_data_ratio'])
        model_batch_size = self.args['policy_batch_size']  - real_batch_size
        
        model_buffer = MOPOBuffer(self.args['buffer_size'])

        for epoch in range(1):
            # collect data
            with torch.no_grad():
                obs = np.concatenate((train_buffer.new_batch_data["state"],train_buffer.new_batch_data["Lambda"]),1)[0:int(self.args['data_collection_per_epoch'])]
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                for t in range(self.args['horizon']):
                    action = self.actor(obs).sample() + torch.ones([len(obs),1]).cuda(device = self.device)
                    action = torch.round(action * 10) / 10 + 0.1
                    obs_action = torch.cat([obs, action], dim=-1)
                    next_obs_dists = transition(obs_action)
                    next_obses = next_obs_dists.sample()
                    rewards = next_obses[:, :, -1:]
                    next_obses = next_obses[:, :, :-1]

                    next_obses_mode = next_obs_dists.mean[:, :, :-1]
                    next_obs_mean = torch.mean(next_obses_mode, dim=0)
                    diff = next_obses_mode - next_obs_mean
                    disagreement_uncertainty = torch.max(torch.norm(diff, dim=-1, keepdim=True), dim=0)[0]
                    aleatoric_uncertainty = torch.max(torch.norm(next_obs_dists.stddev, dim=-1, keepdim=True), dim=0)[0]
                    uncertainty = disagreement_uncertainty if self.args['uncertainty_mode'] == 'disagreement' else aleatoric_uncertainty

                    model_indexes = np.random.randint(0, next_obses.shape[0], size=(obs.shape[0]))
                    next_obs = next_obses[model_indexes, np.arange(obs.shape[0])]
                    reward = rewards[model_indexes, np.arange(obs.shape[0])]
                    
                    print('average reward:', reward.mean().item())
                    print('average uncertainty:', uncertainty.mean().item())

                    penalized_reward = reward - self.args['lam'] * uncertainty
                    dones = torch.zeros_like(reward)

                    batch_data = Batch({
                        "obs" : obs.cpu(),
                        "act" : action.cpu(),
                        "rew" : penalized_reward.cpu(),
                        "done" : dones.cpu(),
                        "obs_next" : next_obs.cpu(),
                    })

                    model_buffer.put(batch_data)

                    obs = next_obs

            # update
            for _ in range(self.args['steps_per_epoch']):
                #batch = train_buffer.sample(real_batch_size)
                model_batch = model_buffer.sample(model_batch_size)
                batch = dict(state=None, action=None, reward=None, done=None, next_state=None)
                a = torch.cat((torch.tensor(torch.from_numpy(train_buffer.new_batch_data["state"][:real_batch_size]), dtype=torch.float32),torch.tensor(torch.from_numpy(train_buffer.new_batch_data["Lambda"][:real_batch_size]), dtype=torch.float32)),1)
                batch['state'] = torch.cat((a,
                                            torch.squeeze(model_batch['obs'])),0).cuda(device=self.device)
                batch['action'] = torch.cat((torch.tensor(torch.from_numpy(train_buffer.new_batch_data["action"][:real_batch_size]), dtype=torch.float32) , model_batch['act']),0).cuda(device=self.device)
                batch['reward'] = torch.cat((torch.tensor(torch.from_numpy(train_buffer.new_batch_data["reward"][:real_batch_size]), dtype=torch.float32), model_batch['rew']),0).cuda(device=self.device)
                batch['done'] = torch.cat((torch.tensor(torch.from_numpy(train_buffer.new_batch_data["done"][:real_batch_size]), dtype=torch.float32) , model_batch['done']),0).cuda(device=self.device)
                b = torch.cat((torch.tensor(torch.from_numpy(train_buffer.new_batch_data["next_state"][:real_batch_size]),
                                            dtype=torch.float32),
                               torch.tensor(torch.from_numpy(train_buffer.new_batch_data["Lambda"][:real_batch_size]),
                                            dtype=torch.float32)), 1)
                batch['next_state'] = torch.cat((b , torch.squeeze(model_batch['obs_next'])),0).cuda(device=self.device)
                #batch.cat_(model_batch)
                #batch.to_torch(device=self.device)

                self._sac_update(batch)

            # res = callback_fn(self.get_policy())
            #
            # res['uncertainty'] = uncertainty.mean().item()
            # res['disagreement_uncertainty'] = disagreement_uncertainty.mean().item()
            # res['aleatoric_uncertainty'] = aleatoric_uncertainty.mean().item()
            # res['reward'] = reward.mean().item()
            # self.log_res(epoch, res)

        return self.get_policy()

    def real_evaluation(self,sess, Number_real_evaluation_users,  Lambda_min, Lambda_max, Lambda_interval, Lambda_size,
                    all_user_come_str, all_user_hongbao_str, all_user_liucun_str, all_hongbao_pre30_str,
                    all_liucun_pre30_str,all_average_liucun_str, all_user_type_str, training_iters, Plot,
                    discount, result_dir):
        Number_days = 30
        max_size_hongbao = 2.1
        max_size_hongbao = max_size_hongbao  # 最大面额红包2.1元
        size_hongbao = int(max_size_hongbao * 10)  # 21个红包
        Number_users = Number_real_evaluation_users
        all_state = np.zeros([Number_users, Number_days, size_hongbao * 2 + 2])
        all_action = np.zeros([Number_users, Number_days, 1])
        all_reward_liucun = np.zeros([Number_users, Number_days, 1])
        all_next_state = np.zeros([Number_users, Number_days, size_hongbao * 2 + 2])
        all_terminal = np.zeros([Number_users, Number_days, 1])
        true_value = np.zeros([Lambda_size, Number_users])
        true_value_cost = np.zeros([Lambda_size, Number_users])

        total_hongbao = []
        total_come = []
        total_convertion = []

        all_user_type2 = np.zeros(Number_users)
        all_user_liucun2 = np.zeros([Number_users, size_hongbao])
        all_user_come2 = np.zeros([Number_users, Number_days])
        all_user_hongbao2 = np.zeros([Number_users, Number_days])
        all_hongbao_pre302 = np.zeros([Number_users, size_hongbao])
        all_liucun_pre302 = np.zeros([Number_users, size_hongbao])
        all_average_liucun2 = np.zeros(Number_users)

        for i in range(Number_users):
            all_user_type2[i] = all_user_type_str[i]
            all_user_liucun2[i, :] = all_user_liucun_str[i]
            all_user_come2[i, :] = all_user_come_str[i]
            all_user_hongbao2[i, :] = all_user_hongbao_str[i]
            all_hongbao_pre302[i, :] = all_hongbao_pre30_str[i]
            all_liucun_pre302[i, :] = all_liucun_pre30_str[i]
            all_average_liucun2[i] = all_average_liucun_str[i]

        for l in range(Lambda_size):

            all_user_type = copy.deepcopy(all_user_type2)
            all_user_liucun = copy.deepcopy(all_user_liucun2)
            all_user_come = copy.deepcopy(all_user_come2)
            all_user_hongbao = copy.deepcopy(all_user_hongbao2)
            all_hongbao_pre30 = copy.deepcopy(all_hongbao_pre302)
            all_liucun_pre30 = copy.deepcopy(all_liucun_pre302)
            all_average_liucun = copy.deepcopy(all_average_liucun2)

            Lambda = Lambda_min + l * Lambda_interval
            Lambda = np.ones([1, 1]) * Lambda
            come_lambda = []
            hongbao_lambda = []
            for user in range(Number_users):
                all_state[user, 0, :] = np.concatenate((np.zeros(1), np.ones(1) * all_user_type[user],
                                                        all_hongbao_pre30[user, :], all_liucun_pre30[user, :]))
            liucun_rate_lambda = np.ones([Number_users, Number_days])
            hongbao_cost_lambda = np.zeros([Number_users, Number_days])
            for d in range(Number_days):
                feed_state_ = np.zeros([Number_users, size_hongbao * 2 + 2])
                new_come = np.zeros(Number_users)
                new_hongbao = np.zeros(Number_users)
                for user in range(Number_users):
                    if all_user_come[user, -1] == 0:  # 前一天没来
                        if np.random.rand() < all_average_liucun[user]:  # 前一天没来今天来了
                            new_come[user] = 1
                            feed_state = np.expand_dims(all_state[user, d, :], axis=0)
                            feed_state_[user, :] = feed_state

                    else:  # 前一天来了
                        if np.random.rand() < all_user_liucun[
                            user, int(all_user_hongbao[user, -1] * 10) - 1]:  # 前一天来了今天也来了
                            all_liucun_pre30[user, int(all_user_hongbao[user, -1] * 10) - 1] += 1
                            new_come[user] = 1
                            feed_state = np.expand_dims(all_state[user, d, :], axis=0)
                            feed_state_[user, :] = feed_state
                            liucun_rate_lambda[user, d] = 0
                        else:
                            liucun_rate_lambda[user, d] = -1
                feed_Lambda_ = np.ones([Number_users, 1]) * Lambda
                obs = np.concatenate((feed_state_, feed_Lambda_), 1)
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                next_action = self.actor(obs).sample() + torch.ones([len(obs),1]).cuda(device = self.device)
                next_action = torch.round(next_action * 10).cpu().numpy()
                #next_action = np.around(next_action, decimals=1) + 0.1
                for user in range(Number_users):
                    if new_come[user] == 1:
                        new_hongbao[user] = (np.squeeze(next_action[user]) + 1) / 10.0
                        hongbao_cost_lambda[user, d] = new_hongbao[user]
                        all_hongbao_pre30[user, int(new_hongbao[user] * 10) - 1] += 1

                        come_lambda.append(new_come[user])
                        hongbao_lambda.append(new_hongbao[user])
                    if d > 0:
                        all_reward_liucun[user, d - 1] = new_come[user] - 1
                        all_next_state[user, d - 1, :] = all_state[user, d, :]
                    if all_user_come[user, 0] == 1:
                        all_hongbao_pre30[user, int(all_user_hongbao[user, 0] * 10) - 1] -= 1
                        if all_user_come[user, 1] == 1:
                            all_liucun_pre30[user, int(all_user_hongbao[user, 0] * 10) - 1] -= 1
                    user_come_temp = all_user_come[user, :]
                    user_come_temp = np.delete(user_come_temp, [0])
                    user_come_temp = np.append(user_come_temp, [new_come[user]])
                    all_user_come[user, :] = user_come_temp
                    user_hongbao_temp = all_user_hongbao[user, :]
                    user_hongbao_temp = np.append(user_hongbao_temp, [new_hongbao[user]])
                    user_hongbao_temp = np.delete(user_hongbao_temp, [0])
                    all_user_hongbao[user, :] = user_hongbao_temp
                    if d != Number_days - 1:
                        all_state[user, d + 1, :] = np.concatenate(
                            (np.ones(1) * (d + 1), np.ones(1) * all_user_type[user], all_hongbao_pre30[user, :],
                             all_liucun_pre30[user, :]))
            for user in range(Number_users):
                liucun_rate_lambda_ = liucun_rate_lambda[user, :]
                hongbao_cost_lambda_ = hongbao_cost_lambda[user, :]
                delete_1 = np.array([1])
                liucun_rate_lambda_ = np.setdiff1d(liucun_rate_lambda_, delete_1)
                true_value[l, user] = sum(np.logspace(0, len(liucun_rate_lambda_) - 1, len(liucun_rate_lambda_),
                                                      base=discount) * liucun_rate_lambda_)
                delete_1 = np.array([0])
                hongbao_cost_lambda_ = np.setdiff1d(hongbao_cost_lambda_, delete_1)
                true_value_cost[l, user] = sum(np.logspace(0, len(hongbao_cost_lambda_) - 1, len(hongbao_cost_lambda_),
                                                           base=discount) * hongbao_cost_lambda_)

            print(
                "[Real Evaluation @learn_count={}], Lambda={}, Monthly_come={}, Monthly_hongbao_cost={}, convertion={}".format(
                    training_iters,
                    round(Lambda_min + l * Lambda_interval, 2),
                    sum(come_lambda),
                    sum(hongbao_lambda),
                    sum(come_lambda) / sum(hongbao_lambda)))
            total_come.append(sum(come_lambda))
            total_hongbao.append(sum(hongbao_lambda))
            total_convertion.append(sum(come_lambda) / sum(hongbao_lambda))

        return true_value, true_value_cost, total_come, total_hongbao, total_convertion


    def _sac_update(self, batch_data):
        obs = batch_data['state']
        action = batch_data['action']
        next_obs = batch_data['next_state']
        reward = batch_data['reward']
        done = batch_data['done']

        # update critic
        obs_action = torch.cat([obs, action], dim=-1)
        _q1 = self.q1(obs_action)
        _q2 = self.q2(obs_action)

        with torch.no_grad():
            next_action_dist = self.actor(next_obs)
            next_action = next_action_dist.sample()
            log_prob = next_action_dist.log_prob(next_action).sum(dim=-1, keepdim=True)
            next_obs_action = torch.cat([next_obs, next_action], dim=-1)
            _target_q1 = self.target_q1(next_obs_action)
            _target_q2 = self.target_q2(next_obs_action)
            alpha = torch.exp(self.log_alpha)
            y = reward + self.args['discount'] * (1 - done) * (torch.min(_target_q1, _target_q2) - alpha * log_prob)

        critic_loss = ((y - _q1) ** 2).mean() + ((y - _q2) ** 2).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # soft target update
        self._sync_weight(self.target_q1, self.q1, soft_target_tau=self.args['soft_target_tau'])
        self._sync_weight(self.target_q2, self.q2, soft_target_tau=self.args['soft_target_tau'])

        if self.args['learnable_alpha']:
            # update alpha
            alpha_loss = - torch.mean(self.log_alpha * (log_prob + self.args['target_entropy']).detach())

            self.log_alpha_optim.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optim.step()

        # update actor
        action_dist = self.actor(obs)
        new_action = action_dist.rsample()
        action_log_prob = action_dist.log_prob(new_action)
        new_obs_action = torch.cat([obs, new_action], dim=-1)
        q = torch.min(self.q1(new_obs_action), self.q2(new_obs_action))
        actor_loss = - q.mean() + torch.exp(self.log_alpha) * action_log_prob.sum(dim=-1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

    def _select_best_indexes(self, metrics, n):
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        selected_indexes = [pairs[i][1] for i in range(n)]
        return selected_indexes

    def _train_transition(self, transition, data, optim):
        #data.to_torch(device=self.device)
        #dist = transition(torch.tensor(torch.from_numpy(np.concatenate((data['state'],data['action']),axis=1)),dtype=torch.float32))
        dist = transition(torch.cat([torch.tensor(torch.from_numpy(data['state']), dtype=torch.float32),
                                     torch.tensor(torch.from_numpy(data['action']), dtype=torch.float32)], dim=1).cuda(device=self.device))
        loss = - dist.log_prob(torch.cat([torch.tensor(torch.from_numpy(data['next_state']), dtype=torch.float32),
                                          torch.tensor(torch.from_numpy(data['reward']), dtype=torch.float32)], dim=1).cuda(device=self.device))
        loss = loss.mean()

        loss = loss + 0.01 * transition.max_logstd.mean() - 0.01 * transition.min_logstd.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        
    def _eval_transition(self, transition, valdata):
        with torch.no_grad():
            #valdata.to_torch(device=self.device)
            dist = transition(torch.cat([torch.tensor(torch.from_numpy(valdata['state']), dtype=torch.float32),
                                          torch.tensor(torch.from_numpy(valdata['action']), dtype=torch.float32)], dim=-1).cuda(device=self.device))
            loss = (dist.mean - torch.cat([torch.tensor(torch.from_numpy(valdata['next_state']), dtype=torch.float32),
                                          torch.tensor(torch.from_numpy(valdata['reward']), dtype=torch.float32)], dim=-1).cuda(device=self.device) ** 2)
            loss = loss.mean(dim=(1,2))
            return list(loss.cpu().numpy())