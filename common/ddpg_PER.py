import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as opt
import os


class DDPG:
    def __init__(self, args):
        self.args = args
        self.gamma = args.gamma
        load_or_not = any([args.load_or_not, args.evaluate])
        self.c_loss = 0
        self.a_loss = 0
        
        # create the network
        self.actor_network = Actor(args)
        self.critic_network = Critic(args)
        
        # build up the target network
        self.actor_target_network = Actor(args)
        self.critic_target_network = Critic(args)
        
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        
        # create the optimizer
        self.actor_optimizer = opt.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optimizer = opt.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        
        # learning rate scheduler
        base_lr_a = 5e-5
        base_lr_c = 5e-5
        lr_a = 5e-4
        lr_c = 5e-3
        self.scheduler_lr_a = opt.lr_scheduler.CyclicLR(self.actor_optimizer,
                                                        base_lr=base_lr_a, max_lr=lr_a, step_size_up=50,
                                                        mode="triangular2", cycle_momentum=False)
        self.scheduler_lr_c = opt.lr_scheduler.CyclicLR(self.critic_optimizer,
                                                        base_lr=base_lr_c, max_lr=lr_c, step_size_up=50,
                                                        mode="triangular2", cycle_momentum=False)
        
        # create the direction for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir+'/'+self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        
        # load model
        load_path = self.args.load_dir+'/'+self.args.load_scenario_name
        actor_pkl = '/actor_params_ep%d.pkl'%self.args.load_episode
        critic_pkl = '/critic_params_ep%d.pkl'%self.args.load_episode
        load_a = load_path+actor_pkl
        load_c = load_path+critic_pkl
        if load_or_not:
            if os.path.exists(load_a):
                self.actor_network.load_state_dict(torch.load(load_a))
                self.critic_network.load_state_dict(torch.load(load_c))
                print('Agent successfully loaded actor_network: {}'.format(load_a))
                print('Agent successfully loaded critic_network: {}'.format(load_c))
    
    # soft update for a single agent
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1-self.args.tau)*target_param.data+self.args.tau*param.data)
        
        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1-self.args.tau)*target_param.data+self.args.tau*param.data)
    
    # choose action
    def choose_action(self, state):
        state = Variable(torch.from_numpy(state))
        # action = self.actor_network.forward(state).detach()
        action = self.actor_network.forward(state)
        new_action = action.data.numpy()
        return new_action
    
    # update the network
    def train(self, state, action, reward, s_next, ISWeights):
        
        state = Variable(torch.from_numpy(state))
        action = Variable(torch.from_numpy(action))
        reward = Variable(torch.from_numpy(reward))
        s_next = Variable(torch.from_numpy(s_next))
        
        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        Q_value = torch.squeeze(self.critic_network.forward(state, action))
        a_next = self.actor_target_network.forward(s_next).detach()
        Q_next = torch.squeeze(self.critic_target_network.forward(s_next, a_next).detach())
        Q_target = torch.squeeze(reward) + self.gamma*Q_next
        # compute critic loss, and update the critic
        TD_error = Q_target - Q_value
        ISWeights_tensor = torch.tensor(ISWeights, dtype=torch.float32)
        weighted_TD_error = torch.mul(TD_error, ISWeights_tensor)
        zero_tensor = torch.zeros(weighted_TD_error.shape)
        critic_loss = F.mse_loss(weighted_TD_error, zero_tensor)
        # original loss
        # critic_loss = F.mse_loss(Q_value, Q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.c_loss = critic_loss.data
        
        # for PER update
        td_error_up = abs(TD_error.detach().numpy())
        
        # ---------------------- optimize actor ----------------------
        pred_a = self.actor_network.forward(state)
        actor_loss = -torch.mean(self.critic_network.forward(state, pred_a))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.a_loss = -actor_loss.data
        
        # soft update
        self._soft_update_target_network()
        
        return td_error_up
    
    def save_model(self, save_episode):
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_params')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path +
                   '/actor_params_ep%d.pkl'%save_episode)
        torch.save(self.critic_network.state_dict(), model_path +
                   '/critic_params_ep%d.pkl'%save_episode)
    
    def lr_scheduler(self):
        lra = self.scheduler_lr_a.get_last_lr()
        lrc = self.scheduler_lr_c.get_last_lr()
        self.scheduler_lr_a.step()
        self.scheduler_lr_c.step()
        return lra, lrc

# define the actor network
class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.obs_shape, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.action_out = nn.Linear(64, args.action_shape)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = torch.tanh(self.action_out(x))  # tanh value section: [-1, 1]
        return action

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.obs_shape+args.action_shape, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.q_out = nn.Linear(128, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value