class EarlyStopping:
    """Stops the training if the validation loss doesn't improve for a given patience."""

    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.do_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.do_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


