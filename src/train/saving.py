import numpy as np
import pickle

# Not for Cross-Validation
def save_model_results(model, x, y, filepath, save_params = False):
    # TODO: implement model parameter saving
    results = model.fit(x, y)
    history = results.history
    # print(history)

    tr_losses = [i['train_rmse'] for i in history]
    t_losses = [i['test_rmse'] for i in history]
    print(tr_losses)

    with open(f'../results/model_{filepath}.pkl', 'wb') as f:
        pickle.dump(model, f)
    np.savez_compressed(f'../results/loss_{filepath}.npz', tr_loss = tr_losses, t_loss = t_losses)

    print(f'saved to {filepath}')