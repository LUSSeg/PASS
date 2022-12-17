import argparse
from src.utils import bool_flag


def getOption():
    parser = argparse.ArgumentParser(description="Implementation of SwAV")

    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                        help="path to dataset repository")
    parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                        help="list of number of crops (example: [2, 6])")
    parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                        help="crops resolutions (example: [224, 96])")
    parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                        help="argument in RandomResizedCrop (example: [0.14, 0.05])")
    parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                        help="argument in RandomResizedCrop (example: [1., 0.14])")

    #########################
    ## swav specific params #
    #########################
    parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                        help="list of crops id used for computing assignments")
    parser.add_argument("--temperature", default=0.1, type=float,
                        help="temperature parameter in training loss")
    parser.add_argument("--epsilon", default=0.05, type=float,
                        help="regularization parameter for Sinkhorn-Knopp algorithm")
    parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                        help="number of iterations in Sinkhorn-Knopp algorithm")
    parser.add_argument("--feat_dim", default=128, type=int,
                        help="feature dimension")
    parser.add_argument("--nmb_prototypes", default=3000, type=int,
                        help="number of prototypes")
    parser.add_argument("--queue_length", type=int, default=0,
                        help="length of the queue (0 for no queue)")
    parser.add_argument("--epoch_queue_starts", type=int, default=15,
                        help="from this epoch, we start using a queue")

    #########################
    #### optim parameters ###
    #########################
    parser.add_argument("--epochs", default=100, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
    parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
    parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                        help="freeze the prototypes during this many iterations from the start")
    parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--start_warmup", default=0, type=float,
                        help="initial warmup learning rate")
    parser.add_argument("--finetune_scale_factor", default=0.5, type=float,
                        help="scale_factor of pseudo-labels in pixel-level finetuning")
    parser.add_argument("--optim", default="sgd", type=str, help="the optimizer for finetuning")
    parser.add_argument("--checkpoint_key", type=str, default='state_dict', help="key of model in checkpoint")


    #########################
    #### dist parameters ###
    #########################
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                        training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                        it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")

    #########################
    #### luss parameters ###
    #########################
    parser.add_argument("--shallow", type=int, default=[3], nargs="+",
                        help="Deep-to-shallow alignment.")
    parser.add_argument("--weights", type=int, default=[1, 1, 1], nargs="+",
                        help="Loss weights for Image-to-Image, Deep-to-Shallow(stage4to3), Deep-to-Shallow(stage4to2)")
    parser.add_argument("--pretrained", default="", type=str, help="path to pretrained weights")
    parser.add_argument("--num_classes", type=int, default=50, help="the number of classes")
    parser.add_argument("--pseudo_path", type=str, default=None, help="the path to generated labels")
    
    #########################
    #### other parameters ###
    #########################
    parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
    parser.add_argument("--hidden_mlp", default=2048, type=int,
                        help="hidden layer dimension in projection head")
    parser.add_argument("--workers", default=10, type=int,
                        help="number of data loading workers")
    parser.add_argument("--checkpoint_freq", type=int, default=25,
                        help="Save the model periodically")
    parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
    parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                        https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
    parser.add_argument("--dump_path", type=str, default=".",
                        help="experiment dump path for checkpoints and log")
    parser.add_argument("--seed", type=int, default=31, help="seed")

    return parser
