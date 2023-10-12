import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from rob831.infrastructure import pytorch_util as ptu
from rob831.policies.base_policy import BasePolicy

from rob831.infrastructure.utils import normalize

class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        action_distribution = self(observation)
        action = action_distribution.sample()  # don't bother with rsample
        return ptu.to_numpy(action)


    # update/train this policy
    def update(self, observations, actions, **kwargs):
        # this raise should be left alone as it is a base class for PG
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            return action_distribution

#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions, advantages, q_values=None, num_traj=1, num_updates=1):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: update the policy using policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
            # is the expectation over collected trajectories of:
            # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
            # by the `forward` method
        # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss
        # HINT4: use self.optimizer to optimize the loss. Remember to
            # 'zero_grad' first
        for update in range(num_updates):      
            start = int((update/num_updates) * len(observations))
            end   = int(((update+1) / num_updates) * len(observations))
            num_traj_minibatch = max(1, int((1/num_updates) * num_traj))
            obs_minibatch = observations[start:end]
            act_minibatch = actions[start:end]
            adv_minibatch = advantages[start:end]
            q_val_minibatch = q_values[start:end]

            # Clear out accumulated gradients
            self.optimizer.zero_grad()

            # Forward pass, get the distribution of action prob.
            act_prob_dist = self.forward(obs_minibatch)

            # Get the log probability for that action(s) that was taken
            # We want to pass our actions through this distribution, this will tell us how probable the actions we took are. 
            # If they have a high advantage value (weight) we will be shifting the weights towards making this action more probable in the future
            log_prob_act = act_prob_dist.log_prob(act_minibatch)

            # Find the loss for the actions, find the negative because we are minimizing the negative log
            # policy_loss = Variable(-torch.sum(log_prob_act.detach() * advantages), requires_grad = True)
            policy_loss = -1/num_traj_minibatch * torch.sum(log_prob_act * adv_minibatch)

            # Backpropegate the loss
            policy_loss.backward()

            # Take an optimization step
            self.optimizer.step()

            if self.nn_baseline:
                ## update the neural network baseline using the q_values as
                ## targets. The q_values should first be normalized to have a mean
                ## of zero and a standard deviation of one.

                # Normalize the q values
                targets = (q_val_minibatch - np.mean(q_val_minibatch)) / np.std(q_val_minibatch)
                targets = ptu.from_numpy(targets)

                # Find the baseline 
                b = self.baseline(obs_minibatch).squeeze()

                # Calculate the loss by sampling from the network and comparing it to the Q value
                loss_baseline = self.baseline_loss.forward(b, targets)

                # Backpropegate the loss and update the parameters
                self.baseline_optimizer.zero_grad()
                loss_baseline.backward()
                self.baseline_optimizer.step()

        # raise NotImplementedError

        train_log = {
            'Training Loss': ptu.to_numpy(policy_loss),
        }
        return train_log

    def run_baseline_prediction(self, observations):
        """
            Helper function that converts `observations` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `observations`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        observations = ptu.from_numpy(observations)
        pred = self.baseline(observations)
        return ptu.to_numpy(pred.squeeze())
