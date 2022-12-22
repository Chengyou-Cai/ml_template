import argparse

class MyConfigArgs():
    
    def __init__(self) -> None:
        self.args = argparse.ArgumentParser()

        # device
        self.args.add_argument('--gpus', type=str, default='0',help="available gpus")
        self.args.add_argument('--which_gpu', type=str, default='cuda:0')
        
        # io
        self.args.add_argument('--io', type=str, default='')

        # hparams
        self.args.add_argument('--rand_seed', type=int, default=3407,help="torch.manual_seed(3407) is all you need")
        self.args.add_argument('--max_epochs', type=int, default=10)
        self.args.add_argument('--num_workers', type=int, default=0)
        self.args.add_argument('--batch_size', type=int, default=128)
        
        self.args.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.args.add_argument('--wd', type=float, default=0.001, help='weight decay')
        self.args.add_argument('--lrd', type=float, default=0.97, help='learning rate decay')
        self.args.add_argument('--clip', type=int, default=3, help='gradient clipping')

    def parse(self):
        return self.args.parse_args() # return config