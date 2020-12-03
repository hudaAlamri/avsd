import argparse
import datetime
import gc
import json
import math
import os
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataloader import VisDialDataset
from encoders import Encoder, LateFusionEncoder
from decoders import Decoder
from utils import process_ranks, scores_to_ranks, get_gt_ranks
import logging



parser = argparse.ArgumentParser()
VisDialDataset.add_cmdline_args(parser)
LateFusionEncoder.add_cmdline_args(parser)
parser.add_argument('--finetune', default=0, type=int)
parser.add_argument('--input_vid', default="data/charades_s3d_mixed_5c_fps_16_num_frames_40_original_scaled", help=".h5 file path for the charades s3d features.")
parser.add_argument('--input_type', default='Q_DH_V', choices=['Q_only','Q_DH',
                                                            'Q_A',
                                                            'Q_I',
                                                            'Q_V',
                                                            'Q_C_I',
                                                            'Q_DH_V',
                                                            'Q_DH_I',
                                                            'Q_V_A',
                                                            'Q_DH_V_A'], help='Specify the inputs')
parser.add_argument_group('Evaluation related arguments')
parser.add_argument('--load_path', default='checkpoints/input_type_Q_DH_V_s3d_mixed_5c_fps_16_num_frames_40_text_encoder_lstm_lr_0.0001_unfreeze_layer_1_finetune_0_use_npy_1_batch_size_12', help='Checkpoint to load path from')
parser.add_argument('--split', default='test', choices=['val', 'test', 'train'], help='Split to evaluate on')
parser.add_argument('--use_gt', action='store_true', help='Whether to use ground truth for retrieving ranks')
parser.add_argument('--batch_size', default=12, type=int, help='Batch size')
parser.add_argument('--gpuid', default=0, type=int, help='GPU id to use')
parser.add_argument('--overfit', action='store_true', help='Use a batch of only 5 examples, useful for debugging')


parser.add_argument_group('Submission related arguments')
parser.add_argument('--save_ranks', action='store_true', help='Whether to save retrieved ranks')
parser.add_argument('--save_path', default='logs/ranks.json', help='Path of json file to save ranks')
parser.add_argument('--crop_only', type=int, default=1,
                            help='random seed')
parser.add_argument('--center_crop', type=int, default=0,
                            help='random seed')
parser.add_argument('--num_frames', type=int, default=40,
                            help='random seed')
parser.add_argument('--video_size', type=int, default=224,
                            help='random seed')
parser.add_argument_group('Evaluation related arguments')

# S3DG parameters and dataloader
parser.add_argument("--use_npy", default=1, help="Uses npy instead of reading from videos")
parser.add_argument("--numpy_path", default="data/charades/num_frames_40/num_frames_40/")
parser.add_argument('--fps', type=int, default=16, help='')
parser.add_argument('--random_flip', type=int, default=0,
                    help='random seed')
parser.add_argument('--video_root', default='data/charades/videos')
parser.add_argument('--unfreeze_layers', default=0, type=int,
                    help="if 1, unfreezes _5 layers, if 2 unfreezes _4 and _5 layers, if 0, unfreezes all layers")
parser.add_argument("--text_encoder", default="lstm",
                    help="lstm or transformer", type=str)

# ----------------------------------------------------------------------------
# input arguments and options
# ----------------------------------------------------------------------------

args = parser.parse_args()
'''
log_path = os.path.join(args.load_path, 'eval_results.log')
logging.basicConfig(filename='eval_results.log')
'''
# seed for reproducibility
torch.manual_seed(1234)
cur = os.getcwd()
os.chdir(args.load_path)
checkpoints = sorted(filter(os.path.isfile, os.listdir('.')), key=os.path.getmtime)
checkpoints = [file for file in checkpoints if file.endswith(".pth")]
logging.info("Evaluate the following checkpoints: %s", args.load_path)
os.chdir(cur)

# set device and default tensor type
if args.gpuid >= 0:
    torch.cuda.manual_seed_all(1234)
    torch.cuda.set_device(args.gpuid)

# set this because only late fusion encoder is supported yet
args.concat_history = True

# ----------------------------------------------------------------------------
# loading dataset wrapping with a dataloader
# ----------------------------------------------------------------------------
dataset = VisDialDataset(args, [args.split])
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=dataset.collate_fn)

# iterations per epoch
setattr(args, 'iter_per_epoch', math.ceil(
    dataset.num_data_points[args.split] / args.batch_size))
print("{} iter per epoch.".format(args.iter_per_epoch))

# ----------------------------------------------------------------------------
# read saved model and args
# ----------------------------------------------------------------------------
for checkpoint in checkpoints:
    print('checkpoint:',checkpoint)
    model_path = os.path.join(args.load_path, checkpoint)
    components = torch.load(model_path)
    model_args = components['model_args']
    model_args.gpuid = args.gpuid
    model_args.batch_size = args.batch_size

    for arg in vars(args):
        print('{:<20}: {}'.format(arg, getattr(args, arg)))

    # ----------------------------------------------------------------------------
    # setup the model
    # ----------------------------------------------------------------------------
    encoder = Encoder(model_args)
    encoder.load_state_dict(components['encoder'])

    decoder = Decoder(model_args, encoder)
    decoder.load_state_dict(components['decoder'])
    print("Loaded model from {}".format(args.load_path))

    if args.gpuid >= 0:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # ----------------------------------------------------------------------------
    # evaluation
    # ----------------------------------------------------------------------------

    print("Evaluation start time: {}".format(
        datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')))
    encoder.eval()
    decoder.eval()


    if args.use_gt:
        # ------------------------------------------------------------------------
        # calculate automatic metrics and finish
        # ------------------------------------------------------------------------
        all_ranks = []
        for i, batch in enumerate(tqdm(dataloader)):
            for key in batch:
                if not isinstance(batch[key], list):
                    batch[key] = Variable(batch[key], volatile=True)
                    if args.gpuid >= 0:
                        batch[key] = batch[key].cuda()

            enc_out = encoder(batch)
            dec_out = decoder(enc_out, batch)
            ranks = scores_to_ranks(dec_out.data)
            gt_ranks = get_gt_ranks(ranks, batch['ans_ind'].data)
            all_ranks.append(gt_ranks)
        all_ranks = torch.cat(all_ranks, 0)
        process_ranks(all_ranks,args.load_path,checkpoint[6:-4])
        gc.collect()
    else:
        # ------------------------------------------------------------------------
        # prepare json for submission
        # ------------------------------------------------------------------------
        ranks_json = []
        for i, batch in enumerate(tqdm(dataloader)):
            for key in batch:
                if not isinstance(batch[key], list):
                    batch[key] = Variable(batch[key], volatile=True)
                    if args.gpuid >= 0:
                        batch[key] = batch[key].cuda()

            enc_out = encoder(batch)
            dec_out = decoder(enc_out, batch)
            ranks = scores_to_ranks(dec_out.data)
            ranks = ranks.view(-1, 10, 100)

            for i in range(len(batch['img_fnames'])):
                # cast into types explicitly to ensure no errors in schema
                if args.split == 'test':
                    ranks_json.append({
                        'image_id': int(batch['img_fnames'][i][-16:-4]),
                        'round_id': int(batch['num_rounds'][i]),
                        'ranks': list(ranks[i][batch['num_rounds'][i] - 1])
                    })
                else:
                    for j in range(batch['num_rounds'][i]):
                        ranks_json.append({
                            'image_id': int(batch['img_fnames'][i][-16:-4]),
                            'round_id': int(j + 1),
                            'ranks': list(ranks[i][j])
                        })
            gc.collect()

    if args.save_ranks:
        print("Writing ranks to {}".format(args.save_path))
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        json.dump(ranks_json, open(args.save_path, 'w'))
