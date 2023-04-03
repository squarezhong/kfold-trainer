# pytorch & sklearn library
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler as Sampler
from util import (get_classification_report, get_confusion_matrix,
                  plot_loss_curve, weights_reset)


# Trainer class
class Trainer:
    def __init__(self, model, dataset, kfold=5, batch_size=128, lr=1e-3, n_epochs=100):
        # using cuda is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataset = dataset
        self.kfold = kfold
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.criterion = nn.CrossEntropyLoss()

        # AdamW = Adam + Weight Decay
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr) # weight_decay=1e-6

        # change the learning rate amid learning to get better performance
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=n_epochs, eta_min=5e-6)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5.0, gamma=0.95)
        
    def train_evaluate(self, model_name) -> None:
        loss_list = []
        kfold = KFold(n_splits=5, shuffle=True)

        # KFold.split() returns the indices of train set and test set each time
        for fold, (train_indices, test_indices) in enumerate(kfold.split(self.dataset)):
            self.model.train()
            # Sample data randomly from dataset with indices
            train_sampler = Sampler(train_indices)
            test_sampler = Sampler(test_indices)
    
            # Use sampler to get dataloader
            train_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_sampler)
            test_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=test_sampler)
    
            # reset weights to avoid weights leaking 
            self.model.zero_grad()
            self.model.apply(weights_reset)
            # clear loss list
            loss_list.clear()
       
            for epoch in range(self.n_epochs):
                current_loss = 0.0
                for inputs, targets in train_loader:
                    targets = targets.type(torch.LongTensor)   # casting to long tensor
                    # use cuda to accelerate the training
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    # zero the gradients
                    self.optimizer.zero_grad()
                    # forward
                    output = self.model(inputs)
                    # get loss
                    loss = self.criterion(output, targets)

                    # propagate back
                    loss.backward()
                    # update optimizer
                    self.optimizer.step()

                    with torch.no_grad():
                        current_loss += loss
                        # update learning rate
                        #self.scheduler.step()

                loss_list.append(current_loss.detach().cpu().numpy())

            # Save the model for each fold
            file_path = './saved_models/{}_fold{}.pth'.format(model_name, str(fold))
            torch.save(self.model.state_dict(), file_path)

            # Evaluate the model using testloader
            pred_list, true_list = [], []
            self.model.eval()
            for inputs, targets in test_loader:
                # use cuda to accelerate the training
                inputs = inputs.to(self.device)
                pred = self.model(inputs)
                # choose the index of max value, dim=1 means reducing 1 dimension
                pred = torch.argmax(pred, dim=1)
                pred_list.extend(pred.detach().cpu().numpy())
                true_list.extend(targets.detach().cpu().numpy())

            # plot loss trend curve
            plot_loss_curve(loss_list, model_name, fold)
            get_classification_report(true_list, pred_list, model_name, fold)
            get_confusion_matrix(true_list, pred_list, model_name, fold)
