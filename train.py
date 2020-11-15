import argparse
import datetime
import gc
import math
import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataloader import VisDialDataset
from encoders import Encoder, LateFusionEncoder
from decoders import Decoder

parser = argparse.ArgumentParser()
VisDialDataset.add_cmdline_args(parser)
LateFusionEncoder.add_cmdline_args(parser)

parser.add_argument_group('Input modalites arguments')
parser.add_argument('-input_type', default='question_dialog_video', choices=['question_only',
                                                                             'question_dialog',
                                                                             'question_audio',
                                                                             'question_image',
                                                                             'question_video',
                                                                             'question_caption_image',
                                                                             'question_dialog_video',
                                                                             'question_dialog_image',
                                                                             'question_video_audio',
                                                                             'question_dialog_video_audio'], help='Specify the inputs')

parser.add_argument_group('Encoder Decoder choice arguments')
parser.add_argument('-encoder', default='lf-ques-im-hist',
                    choices=['lf-ques-im-hist'], help='Encoder to use for training')
parser.add_argument('-concat_history', default=True,
                    help='True for lf encoding')
parser.add_argument('-decoder', default='disc',
                    choices=['disc'], help='Decoder to use for training')

parser.add_argument_group('Optimization related arguments')
parser.add_argument('-num_epochs', default=20, type=int, help='Epochs')
parser.add_argument('-batch_size', default=12, type=int, help='Batch size')
parser.add_argument('-lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('-lr_decay_rate', default=0.9997592083,
                    type=float, help='Decay for lr')
parser.add_argument('-min_lr', default=5e-5, type=float,
                    help='Minimum learning rate')
parser.add_argument('-weight_init', default='xavier',
                    choices=['xavier', 'kaiming'], help='Weight initialization strategy')
parser.add_argument('-weight_decay', default=0.00075,
                    help='Weight decay for l2 regularization')
parser.add_argument('-overfit', action='store_true',
                    help='Overfit on 5 examples, meant for debugging')
parser.add_argument('-gpuid', default=0, type=int, help='GPU id to use')

parser.add_argument_group('Checkpointing related arguments')
parser.add_argument('-load_path', default='',
                    help='Checkpoint to load path from')
parser.add_argument('-save_path', default='checkpoints/',
                    help='Path to save checkpoints')
parser.add_argument('-save_step', default=2, type=int,
                    help='Save checkpoint after every save_step epochs')
parser.add_argument(
    '--input_vid', default="./data/charades/charades_s3d_mixed_5c_fps_16_num_frames_40_original_scaled", help=".h5 file path for the charades s3d features.")
parser.add_argument('--finetune', default=0, type=int,
                    help="When set true, the model finetunes the s3dg model for video")
# S3DG parameters and dataloader
parser.add_argument('--num_frames', type=int, default=40,
                    help='random seed')
parser.add_argument('--video_size', type=int, default=224,
                    help='random seed')
parser.add_argument('--fps', type=int, default=16, help='')
parser.add_argument('--crop_only', type=int, default=1,
                    help='random seed')
parser.add_argument('--center_crop', type=int, default=0,
                    help='random seed')
parser.add_argument('--random_flip', type=int, default=0,
                    help='random seed')
parser.add_argument('--video_root', default='./data/charades/videos')
parser.add_argument('--unfreeze_layers', default=0, type=int,
                    help="if 1, unfreezes _5 layers, if 2 unfreezes _4 and _5 layers, if 0, unfreezes all layers")
parser.add_argument("--text_encoder", default="lstm",
                    help="lstm or transformer", type=str)
# ----------------------------------------------------------------------------
# input arguments and options
# ----------------------------------------------------------------------------

args = parser.parse_args()
start_time = datetime.datetime.strftime(
    datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')
if args.save_path == 'checkpoints/':
    # args.save_path += start_time
    args.save_path += 's3d_mixed_5c_fps_{0}_num_frames_{1}_text_encoder_{2}_lr_{3}_unfreeze_layer_{4}_finetune_{5}'.format(
        args.fps, args.num_frames, args.text_encoder, args.lr, args.unfreeze_layers, finetune)

# seed for reproducibility
torch.manual_seed(1234)

# set device and default tensor type
if args.gpuid >= 0:
    torch.cuda.manual_seed_all(1234)
    torch.cuda.set_device(args.gpuid)

# transfer all options to model
model_args = args

# ----------------------------------------------------------------------------
# read saved model and args
# ----------------------------------------------------------------------------

if args.load_path != '':
    components = torch.load(args.load_path)
    model_args = components['model_args']
    model_args.gpuid = args.gpuid
    model_args.batch_size = args.batch_size

    # this is required by dataloader
    args.img_norm = model_args.img_norm

for arg in vars(args):
    print('{:<20}: {}'.format(arg, getattr(args, arg)))

# ----------------------------------------------------------------------------
# loading dataset wrapping with a dataloader
# ----------------------------------------------------------------------------

dataset = VisDialDataset(args, ['train'])
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=dataset.collate_fn)

dataset_val = VisDialDataset(args, ['val'])
dataloader_val = DataLoader(dataset_val,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=dataset.collate_fn)

# ----------------------------------------------------------------------------
# setting model args
# ----------------------------------------------------------------------------

# transfer some useful args from dataloader to model
for key in {'num_data_points', 'vocab_size', 'max_ques_count',
            'max_ques_len', 'max_ans_len'}:
    setattr(model_args, key, getattr(dataset, key))

# iterations per epoch
setattr(args, 'iter_per_epoch', math.ceil(
    dataset.num_data_points['train'] / args.batch_size))
print("{} iter per epoch.".format(args.iter_per_epoch))

# ----------------------------------------------------------------------------
# setup the model
# ----------------------------------------------------------------------------

encoder = Encoder(model_args)
decoder = Decoder(model_args, encoder)
total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print("Total number of encoder params {0}".format(total_params))
if args.finetune:
    total_params = sum(p.numel()
                       for p in encoder.video_embed.parameters() if p.requires_grad)
    print("Total number of s3dg params {0}".format(total_params))
optimizer = optim.Adam(list(encoder.parameters(
)) + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(
    optimizer, step_size=1, gamma=args.lr_decay_rate)

if args.load_path != '':
    encoder.load_state_dict(components['encoder'])
    decoder.load_state_dict(components['decoder'])
    print("Loaded model from {}".format(args.load_path))
print("Encoder: {}".format(args.encoder))
print("Decoder: {}".format(args.decoder))

if args.gpuid >= 0:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    criterion = criterion.cuda()

# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------

encoder.train()
decoder.train()
os.makedirs(args.save_path, exist_ok=True)
with open(os.path.join(args.save_path, "args_{0}.txt".format(start_time)), "w") as f:
    f.write(str(args))
f.close()wf

running_loss = 0.0
train_begin = datetime.datetime.utcnow()
print("Training start time: {}".format(
    datetime.datetime.strftime(train_begin, '%d-%b-%Y-%H:%M:%S')))

log_loss = []
for epoch in range(1, model_args.num_epochs + 1):
    for i, batch in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        for key in batch:
            if not isinstance(batch[key], list):
                batch[key] = Variable(batch[key])
                if args.gpuid >= 0:
                    batch[key] = batch[key].cuda()

        # --------------------------------------------------------------------
        # forward-backward pass and optimizer step
        # --------------------------------------------------------------------
        enc_out = encoder(batch)
        dec_out = decoder(enc_out, batch)

        cur_loss = criterion(dec_out, batch['ans_ind'].view(-1))
        cur_loss.backward()

        optimizer.step()
        gc.collect()

        # --------------------------------------------------------------------
        # update running loss and decay learning rates
        # --------------------------------------------------------------------
        train_loss = cur_loss.item()
        if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * cur_loss.item()
        else:
            running_loss = cur_loss.item()

        if optimizer.param_groups[0]['lr'] > args.min_lr:
            scheduler.step()

        # --------------------------------------------------------------------
        # print after every few iterations
        # --------------------------------------------------------------------
        if i % 500 == 0:
            validation_losses = []
            for _, val_batch in tqdm(enumerate(dataloader_val)):
                for key in val_batch:
                    if not isinstance(val_batch[key], list):
                        val_batch[key] = Variable(val_batch[key])
                        if args.gpuid >= 0:
                            val_batch[key] = val_batch[key].cuda()
                enc_out = encoder(val_batch)
                dec_out = decoder(enc_out, val_batch)

                cur_loss = criterion(dec_out, val_batch['ans_ind'].view(-1))
                validation_losses.append(cur_loss.item())

            validation_loss = np.mean(validation_losses)

            iteration = (epoch - 1) * args.iter_per_epoch + i

            log_loss.append((epoch,
                             iteration,
                             running_loss,
                             train_loss,
                             validation_loss,
                             optimizer.param_groups[0]['lr']))

            # print current time, running average, learning rate, iteration, epoch
            print("[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][val loss: {:6f}][lr: {:7f}]".format(
                datetime.datetime.utcnow() - train_begin, epoch,
                iteration, running_loss, validation_loss,
                optimizer.param_groups[0]['lr']))

    # ------------------------------------------------------------------------
    # save checkpoints and final model
    # ------------------------------------------------------------------------
    if epoch % args.save_step == 0:
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': encoder.args
        }, os.path.join(args.save_path, 'model_epoch_{}.pth'.format(epoch)))

torch.save({
    'encoder': encoder.state_dict(),
    'decoder': decoder.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': encoder.args
}, os.path.join(args.save_path, 'model_final.pth'))

np.save(os.path.join(args.save_path, 'log_loss'), log_loss)
