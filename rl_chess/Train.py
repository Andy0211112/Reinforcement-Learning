import torch
from Config import Config
from datetime import datetime
from Datahelper import get_game_data_filenames, load_data_from_file

config = Config()


class Trainer:
    def __init__(self, config=config):
        self.config = config
        self.device = self.config.device
        self.epochs = self.config.trainer.epochs
        self.batch_size = self.config.trainer.batch_size

    def dataloader(self, state_ary, policy_ary, value_ary, batch_size):
        states = []
        policies = []
        values = []
        n = state_ary.shape[0]
        for i in range(n//batch_size):
            states.append(torch.tensor(
                state_ary[i*batch_size:(i+1)*batch_size], dtype=torch.float32).to(self.device))
            policies.append(torch.tensor(
                policy_ary[i*batch_size:(i+1)*batch_size], dtype=torch.float32).to(self.device))
            values.append(torch.tensor(
                value_ary[i*batch_size:(i+1)*batch_size], dtype=torch.float32).to(self.device))
        if n % batch_size != 0:
            states.append(torch.tensor(
                state_ary[n//batch_size*batch_size:], dtype=torch.float32).to(self.device))
            policies.append(torch.tensor(
                policy_ary[n//batch_size*batch_size:], dtype=torch.float32).to(self.device))
            values.append(torch.tensor(
                value_ary[n//batch_size*batch_size:], dtype=torch.float32).to(self.device))
        return states, policies, values

    def train_from_data(self, agent, play_data_filename_tmpl):
        filenames = get_game_data_filenames(
            self.config.play_data_dir, play_data_filename_tmpl)

        print(filenames)
        print(len(filenames))
        steps = 0
        D = datetime.today()
        print(f'Training agent {agent.name}')
        while len(filenames) > 0:
            states, policies, values = ([], [], [])
            for _ in range(15):
                if not filenames:
                    break
                filename = filenames.pop()
                print(f'load data from {filename}')
                state_ary, policy_ary, value_ary = load_data_from_file(
                    filename)

                temp = self.dataloader(
                    state_ary, policy_ary, value_ary, self.batch_size)
                states, policies, values = states + \
                    temp[0], policies+temp[1], values+temp[2]
                del temp, state_ary, policy_ary, value_ary
            print(f'num of batch: {len(values)}')
            for epoch in range(self.epochs):
                running_loss = 0
                for state, policy, value in zip(states, policies, values):

                    predicted_policies, predicted_values = agent.model(state)

                    loss = agent.criterion(
                        predicted_values, value, predicted_policies, policy)

                    agent.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.model.parameters(), max_norm=1.0)
                    agent.optimizer.step()
                    running_loss += loss.item()
                if (epoch+1) % 1 == 0:
                    print(
                        f'Epoch {epoch+1}/{self.epochs}, Loss: {running_loss/len(values) :.4f}')
                if (epoch+1) % 100 == 0:
                    agent.save_model(self.config.model_dir,
                                     f'train_{D.month:02}{D.day:02}{D.hour:02}{D.minute:02}{steps:04}.pt')
                steps += 1
            agent.save_model(self.config.model_dir,
                             f'train_{D.month:02}{D.day:02}{D.hour:02}{D.minute:02}{steps:04}.pt')
