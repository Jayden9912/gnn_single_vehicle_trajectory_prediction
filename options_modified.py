import os
import argparse

class NN_Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="training options")

        self.parser.add_argument("--batch_size",
                                 type = int,
                                 help = "number of batch size",
                                 default = 16)

        self.parser.add_argument("--learning_rates",
                                 type = int,
                                 help = "learning rates",
                                 default = 0.001)

        self.parser.add_argument("--scheduler_step_size",
                                 type = int,
                                 help = "scheduler step size",
                                 default = 60000*5)

        self.parser.add_argument("--num_epoch",
                                 type = int,
                                 help = "num_epoch",
                                 default = 50)

        # self.parser.add_argument("--log_frequency",
        #                          type=int,
        #                          help="number of batches between each tensorboard log",
        #                          default=50)
        
        #embedding
        self.parser.add_argument("--emb_output_features",
                                 type = int,
                                 help = "number of output features for embedding layer",
                                 default = 16)
                                 
        self.parser.add_argument("--enc_emb_output_features",
                                 type = int,
                                 help = "number of output features for embedding layer",
                                 default = 32)
        #encoder
        self.parser.add_argument("--enc_output_features",
                                 type = int,
                                 help = "number of output features for encoder output(neighbouring vehicles)",
                                 default = 32)

        self.parser.add_argument("--ego_enc_output_features",
                                 type = int,
                                 help = "number of output features for encoder output(ego vehicles)",
                                 default = 64)
        
        #conv
        self.parser.add_argument("--conv_output_features",
                                 type = int,
                                 help = "number of output features",
                                 default = 32)
        
        self.parser.add_argument("--conv_head",
                                 type = int,
                                 help = "number of head for GAT encoder",
                                 default = 4)
        
        #decoder
        self.parser.add_argument("--dec_num_layers",
                                 type = int,
                                 help = "number of layers for decoder",
                                 default = 2)

        self.parser.add_argument("--dec_input_features",
                                 type = int,
                                 help = "number of features for decoder",
                                 default = 64)
        #other
        self.parser.add_argument("--no_cuda",
                                 action = "store_true",
                                 help = "no_cuda if set")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options