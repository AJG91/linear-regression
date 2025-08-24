
import torch as tc
import torch.nn as nn

from plots import plot_true_vs_pred

class BuildLinearLayer():
    def __init__(self, num_features, lr, bias=True):
        self.model = nn.Linear(in_features=num_features[0], out_features=num_features[1], bias=bias)
        self.loss_fn = nn.MSELoss()
        self.optimizer = tc.optim.SGD(self.model.parameters(), lr=lr)

    def concat_nonlinear(self, X):
        return tc.cat([X, X**2], dim=1)
        
    def train(self, X, y, epochs, atol=1e-8):
        model = self.model
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        
        prev_loss = 0
        X_feat = self.concat_nonlinear(X)
        
        for i, epoch_i in enumerate(range(epochs)):
            y_pred = model(X_feat)
            loss = loss_fn(y_pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if (i + 1) % 100 == 0:
                print(f'epoch = {i + 1}, loss = {loss}')
                plot_true_vs_pred(X, y, tc.squeeze(y_pred).detach())
        
            if abs(prev_loss - loss) < atol:
                print('Training done')
                print(f'Epoch = {i + 1}, loss = {loss}')
                print(f'Residual: {abs(prev_loss - loss)} < {atol}')
                plot_true_vs_pred(X, y, tc.squeeze(y_pred).detach())
                break
    
            prev_loss = loss

        return None
            
    def predict(self, X):
        
        with tc.no_grad():
            y = self.model(self.concat_nonlinear(X))
            
        return tc.squeeze(y.detach())

