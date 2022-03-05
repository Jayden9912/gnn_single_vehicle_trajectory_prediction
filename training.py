from trainer import trainer
from options_modified import NN_Options

options = NN_Options()
opts = options.parse()

if __name__ == "__main__":
    training = trainer(opts)
    training.train()

