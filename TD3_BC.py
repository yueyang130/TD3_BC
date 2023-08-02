import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Clip(nn.Module):
    def __init__(self, min_val, max_val):
        super(Clip, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)


class WeightNorm(nn.Module):
    append_g = '_g'
    append_v = '_v'

    def __init__(self, module, weights):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            g = torch.norm(w)
            v = w/g.expand_as(w)
            g = Parameter(g.data)
            v = Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v

            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)

    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v*(g/torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, layernorm, hidden_dim=256, last_act_bound=1, dropout_prob=0):
        super(Critic, self).__init__()
  
        # # Q1 architecture
        # self.l1 = nn.Linear(state_dim + action_dim, 256)
        # self.l2 = nn.Linear(256, 256)
        # self.l3 = nn.Linear(256, 1)

        # # Q2 architecture
        # self.l4 = nn.Linear(state_dim + action_dim, 256)
        # self.l5 = nn.Linear(256, 256)
        # self.l6 = nn.Linear(256, 1)
        if layernorm in [0,1,6]:
            if layernorm == 0:
                normalization = nn.Identity()
            if layernorm == 1:
                normalization = nn.LayerNorm(hidden_dim)
            if layernorm == 6:
                normalization = nn.BatchNorm1d(hidden_dim)
            
            
            self.q1 = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                normalization,
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, hidden_dim),
                normalization,
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, 1)
            )
            self.q2 = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                normalization,
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, hidden_dim),
                normalization,
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, 1)
            )
        elif layernorm == 4:        
            self.q1 = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                Clip(0, last_act_bound),
                nn.Linear(hidden_dim, 1)
            )
            self.q2 = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                Clip(0, last_act_bound),
                nn.Linear(hidden_dim, 1)
            )
        elif layernorm == 5:        
            self.q1 = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                Clip(0, last_act_bound),
            )
            self.q2 = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                Clip(0, last_act_bound),
            )
        else:
            if layernorm == 2:
                normalization = partial(WeightNorm, weights=['weight'])
            if layernorm == 3:
                normalization = nn.utils.spectral_norm
                
            self.q1 = nn.Sequential(
                normalization(nn.Linear(state_dim + action_dim, hidden_dim)),
                nn.ReLU(),
                normalization(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                normalization(nn.Linear(hidden_dim, 1))
            )
            self.q2 = nn.Sequential(
                normalization(nn.Linear(state_dim + action_dim, hidden_dim)),
                nn.ReLU(),
                normalization(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                normalization(nn.Linear(hidden_dim, 1))
            )
            
        


    # def forward(self, state, action):
    #     sa = torch.cat([state, action], 1)
    #     return self.q1(sa), self.q2(sa)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        # Passing the input through all layers except the last one to get the penultimate output
        penultimate_q1 = sa
        for layer in self.q1[:-1]:
            penultimate_q1 = layer(penultimate_q1)

        penultimate_q2 = sa
        for layer in self.q2[:-1]:
            penultimate_q2 = layer(penultimate_q2)

        # Passing the penultimate output through the last layer to get the final output
        final_q1 = self.q1[-1](penultimate_q1)
        final_q2 = self.q2[-1](penultimate_q2)

        return final_q1, final_q2, penultimate_q1, penultimate_q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)


class TD3_BC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        reweight_eval,
        reweight_improve,
        reweight_constraint,
        clip_constraint,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
        bc_coef=1.0,
        qf_layer_norm=False,
        last_act_bound=1.0,
        weight_decay=0,
        dropout_prob=0,
        dr3_coef=0,
        **kwargs,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, qf_layer_norm, last_act_bound=last_act_bound, dropout_prob=dropout_prob).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, weight_decay=weight_decay)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.bc_coef = bc_coef
        self.dr3_coef = dr3_coef

        self.reweight_eval = reweight_eval
        self.reweight_improve = reweight_improve
        self.reweight_constraint = reweight_constraint
        self.clip_constraint = clip_constraint

        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, two_sampler=False):
        self.total_it += 1
        # Sample replay buffer 
        if two_sampler:
            state, action, next_state, reward, not_done, weight = replay_buffer.sample(uniform=True)
        else:
            state, action, next_state, reward, not_done, weight = replay_buffer.sample()

        # Select action according to policy and add clipped noise
        noise = (
            torch.randn_like(action) * self.policy_noise
        ).clamp(-self.noise_clip, self.noise_clip)
        
        next_action = (
            self.actor_target(next_state) + noise
        ).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2, _, _ = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2).detach()
        target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2, phi_q1, phi_q2 = self.critic(state, action)
        

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q, reduction='none') + F.mse_loss(current_Q2, target_Q, reduction='none')
        if self.reweight_eval:
            critic_loss *= weight
        
        if self.dr3_coef > 0:
            _, _, phi_q1_prime, phi_q2_prime = self.critic(next_state, next_action)
            dr3_loss = torch.sum(phi_q1_prime * phi_q1, dim=1).mean() + torch.sum(phi_q2_prime * phi_q2, dim=1).mean()
            critic_loss += dr3_loss * self.dr3_coef
        
        critic_loss = critic_loss.mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        actor_infos = {}
        if self.total_it % self.policy_freq == 0:
            if two_sampler:
                state, action, next_state, reward, not_done, weight = replay_buffer.sample()
            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha/Q.abs().mean().detach()

            # policy improvement
            actor_loss = Q
            if self.reweight_improve:
                actor_loss *= weight
            actor_loss = actor_loss.mean()
            # policy constraint
            constraint_loss = F.mse_loss(pi, action, reduction='none') 
            if self.reweight_constraint:
                if self.clip_constraint == 1:
                    c_weight = torch.clamp(weight, 1.0)
                elif self.clip_constraint == 2:
                    c_weight = copy.deepcopy(weight)
                    c_weight[weight < 1] = torch.sqrt(weight[weight < 1])
                else:
                    c_weight = weight
                constraint_loss *= c_weight
            constraint_loss = constraint_loss.mean()
            actor_loss = -lmbda * actor_loss + constraint_loss * self.bc_coef
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # log actor training
            actor_infos = {
                "critic_loss": critic_loss.mean().cpu(),
                "actor_loss": actor_loss.mean().cpu(),
                "constraint_loss": constraint_loss.mean().cpu(),
                "dr3_loss": dr3_loss.mean().cpu(),
                "lambda": lmbda.cpu(), 
            }

        def flatten_parameters(model):
            return torch.cat([param.view(-1) for param in model.parameters()])

        def model_weights_norm(m1):
            m1_flat = flatten_parameters(m1)
            m1_norm = torch.norm(m1_flat, p=2)
            return m1_norm.item()

        return {
            "Q1": current_Q1.mean().cpu(),
            "Q2": current_Q2.mean().cpu(),
            "Q1_norm": model_weights_norm(self.critic.q1),
            "Q2_norm": model_weights_norm(self.critic.q2),
            **actor_infos,
        }

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

