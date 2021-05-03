import os
import sys
import numpy as np

# special case CUDA 9
import ctypes
ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)

import tensorflow as tf
import model
import utils


from config import Config

import pdb
# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--data_path', default='/Erfan/Audio/voxceleb1/wav', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)


args = parser.parse_args()
cfg = Config()

def main():

    # gpu configuration
    utils.test_GPU(args)

    # ==================================
    #       Get Train/Val.
    # ==================================
    print('==> calculating test({}) data lists...'.format(args.test_type))
    
    try:
        verify_list = np.loadtxt(cfg.test_map[args.test_type], str)
    except:
        raise IOError('==> test loading failed.')

    verify_lb = np.array([int(i[0]) for i in verify_list])
    list1 = np.array([os.path.join(args.data_path, i[1]) for i in verify_list])
    list2 = np.array([os.path.join(args.data_path, i[2]) for i in verify_list])

    total_list = np.concatenate((list1, list2))
    unique_list = np.unique(total_list) # number of unique wav/ audio files

    print(unique_list.shape)


    # ==================================
    #       Get Model
    # ==================================
    # construct the data generator.

    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)
    
    print(network_eval)

    # ==> load pre-trained model ???
    if args.resume:
        # ==> get real_model from arguments input,
        # load the model if the imag_model == real_model.
        if os.path.isfile(args.resume):
            network_eval.load_weights(os.path.join(args.resume), by_name=True)
            result_path = set_result_path(args)
            print('==> successfully loading model {}.'.format(args.resume))
        else:
            raise IOError("==> no checkpoint found at '{}'".format(args.resume))
    else:
        pass
        # raise IOError('==> please type in the model to load')

    print('==> start testing.')

    # The feature extraction process has to be done sample-by-sample,
    # because each sample is of different lengths.
    total_length = len(unique_list)
    feats, scores, labels = [], [], []
    for c, ID in enumerate(unique_list):
        if c % 50 == 0: print('Finish extracting features for {}/{}th wav.'.format(c, total_length))
        specs = utils.load_data(ID, win_length=params['win_length'], sr=params['sampling_rate'],
                             hop_length=params['hop_length'], n_fft=params['nfft'],
                             spec_len=params['spec_len'], mode='eval')
        specs = np.expand_dims(np.expand_dims(specs, 0), -1)
    
        v = network_eval.predict(specs)
        feats += [v]
    
    feats = np.array(feats)

    # ==> compute the pair-wise similarity.
    for c, (p1, p2) in enumerate(zip(list1, list2)):
        ind1 = np.where(unique_list == p1)[0][0]
        ind2 = np.where(unique_list == p2)[0][0]

        v1 = feats[ind1, 0]
        v2 = feats[ind2, 0]

        scores += [np.sum(v1*v2)]
        labels += [verify_lb[c]]
        print('scores : {}, gt : {}'.format(scores[-1], verify_lb[c]))

    scores = np.array(scores)
    labels = np.array(labels)

    np.save(os.path.join(cfg.result_path, 'prediction_scores.npy'), scores)
    np.save(os.path.join(cfg.result_path, 'groundtruth_labels.npy'), labels)

    eer, thresh = utils.calculate_eer(labels, scores)
    print('==> model : {}, EER: {}'.format(args.resume, eer))



if __name__ == "__main__":
    main()