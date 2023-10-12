import numpy as np

# Not for Cross-Validation
def save_model_results(model, x, y, filepath, save_params = False):
    # TODO: implement model parameter saving
    results = model.fit(x, y)
    history = results.history

    tr_losses = [i['train_loss'] for i in history]
    t_losses = [i['valid_loss'] for i in history]

    np.savez_compressed(filepath, tr_loss = tr_losses, t_loss = t_losses)

    print(f'saved to {filepath}')