"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
import sys
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import trange

import utils.mnist_input as mnist_input
from mnist_eval import evaluate 
import models.small_cnn as small_cnn
from attacks.spatial_attack import SpatialAttack
import utils.utilities as utilities

def train(config):
    # seeding randomness
    tf.set_random_seed(config.training.tf_random_seed)
    np.random.seed(config.training.np_random_seed)

    # Setting up training parameters
    max_num_training_steps = config.training.max_num_training_steps
    step_size_schedule = config.training.step_size_schedule
    weight_decay = config.training.weight_decay
    momentum = config.training.momentum
    batch_size = config.training.batch_size
    adversarial_training = config.training.adversarial_training
    eval_during_training = config.training.eval_during_training
    LAMBDA = float(config.training.unsupervised_lambda)
    if eval_during_training:
        num_eval_steps = config.training.num_eval_steps

    use_kl = config.attack.use_kl
    # Setting up output parameters
    num_output_steps = config.training.num_output_steps
    num_summary_steps = config.training.num_summary_steps
    num_checkpoint_steps = config.training.num_checkpoint_steps

    # Setting up the data and the model
    data_path = config.data.data_path
    raw_cifar = mnist_input.MNISTData(data_path, config.training.partial, config.training.unlabel)
    global_step = tf.train.get_or_create_global_step()
    model = small_cnn.Model(config.model)


    # Setting up the optimizer
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values)
    total_loss = model.mean_xent + weight_decay * model.weight_decay_loss

    if use_kl:
        total_loss += model.mean_kl

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_step = optimizer.minimize( total_loss, global_step=global_step)

    # Set up adversary
    attack = SpatialAttack(model, config.attack)

    # Setting up the Tensorboard and checkpoint outputs
    model_dir = config.model.output_dir
    if eval_during_training:
        eval_dir = os.path.join(model_dir, 'eval')
        if not os.path.exists(eval_dir):
          os.makedirs(eval_dir)

    # We add accuracy and xent twice so we can easily make three types of
    # comparisons in Tensorboard:
    # - train vs eval (for a single run)
    # - train of different runs
    # - eval of different runs

    saver = tf.train.Saver(max_to_keep=30)

    tf.summary.scalar('accuracy_adv_train', model.accuracy, collections=['adv'])
    tf.summary.scalar('accuracy_adv', model.accuracy, collections=['adv'])
    tf.summary.scalar('xent_adv_train', model.xent / batch_size,
                                                        collections=['adv'])
    tf.summary.scalar('xent_adv', model.xent / batch_size, collections=['adv'])
    tf.summary.image('images_adv_train', model.x_image, collections=['adv'])
    adv_summaries = tf.summary.merge_all('adv')

    tf.summary.scalar('accuracy_nat_train', model.accuracy, collections=['nat'])
    tf.summary.scalar('accuracy_nat', model.accuracy, collections = ['nat'])
    tf.summary.scalar('xent_nat_train', model.xent / batch_size,
                                                        collections=['nat'])
    tf.summary.scalar('xent_nat', model.xent / batch_size, collections=['nat'])
    tf.summary.image('images_nat_train', model.x_image, collections=['nat'])
    tf.summary.scalar('learning_rate', learning_rate, collections=['nat'])
    nat_summaries = tf.summary.merge_all('nat')

    with tf.Session() as sess:

      # initialize data augmentation
      if config.training.data_augmentation:
          cifar = mnist_input.AugmentedMNISTData(raw_cifar, sess)
      else:
          cifar = raw_cifar

      # Initialize the summary writer, global variables, and our time counter.
      summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
      if eval_during_training:
          eval_summary_writer = tf.summary.FileWriter(eval_dir)

      sess.run(tf.global_variables_initializer())
      training_time = 0.0

      len_label, len_unlabel = float(raw_cifar.train_data.n), float(raw_cifar.unlabeled_data.n)

      p = len_label / (len_label + len_unlabel) 
      
      # Main training loop
      for ii in range(max_num_training_steps+1):
        x_batch, y_batch = cifar.train_data.get_next_batch(int(batch_size * p),
                                                           multiple_passes=True)
        if config.training.unsupervised == 'semi':
            
            
            x_unlabel, _ = cifar.unlabeled_data.get_next_batch(int(batch_size * (1-p)),
                                                               multiple_passes=True)
            x_mix = np.concatenate((x_batch, x_unlabel), axis=0)
        
            y_prediction = sess.run(model.softmax if use_kl else model.predictions, feed_dict={model.x_input: x_mix,\
                                                                model.y_input: np.concatenate((y_batch, y_batch), axis=0),\
                                                                model.transform: np.zeros([len(x_mix), 3]),\
                                                                model.weights: [1. for i in range(len(x_mix))],\
                                                                model.is_training: False})
        elif config.training.unsupervised == 'nosemi':
            
            x_mix = x_batch
            y_prediction = sess.run(model.softmax if use_kl else model.predictions, feed_dict={model.x_input: x_mix,\
                                                                model.y_input: y_batch,\
                                                                model.transform: np.zeros([len(x_mix), 3]),\
                                                                model.weights: [1. for i in range(len(x_mix))],\
                                                                model.is_training: False})
        noop_trans = np.zeros([len(x_batch), 3])
        # Compute Adversarial Perturbations
        if adversarial_training:
            start = timer()
            if 'semi' in config.training.unsupervised:
                x_batch_adv, adv_trans = attack.perturb(x_mix, y_prediction, sess)
            else:
                x_batch_adv, adv_trans = attack.perturb(x_batch, y_batch, sess)
            end = timer()
            training_time += end - start
        else:
            x_batch_adv, adv_trans = x_batch, noop_trans

        nat_dict = {model.x_input: x_batch,
                    model.y_input: y_batch,
                    model.transform: noop_trans,
                    model.weights: [1. for i in range(len(x_batch))],
                    model.is_training: False}

        if use_kl:
            adv_dict = {model.x_input: np.concatenate((x_batch, x_batch_adv), axis=0),
                        model.y_input: np.concatenate((y_batch, np.zeros(len(x_batch_adv))), axis=0),
                        model.y_pred_input: np.concatenate((np.zeros((len(x_batch), 10)), y_prediction), axis=0),
                        model.transform: np.concatenate((np.zeros([len(x_batch), 3]), adv_trans)),
                        model.weights: [1. if i < len(x_batch) else 0 for i in range((len(x_batch) + len(x_batch_adv)))],
                        model.kl_weights: [0. if i < len(x_batch) else LAMBDA for i in range((len(x_batch) + len(x_batch_adv)))],
                        model.is_training: False}
        else:
            adv_dict = {model.x_input: np.concatenate((x_batch, x_batch_adv), axis=0) if 'semi' in config.training.unsupervised else x_batch_adv,
                        model.y_input: np.concatenate((y_batch, y_prediction), axis=0) if 'semi' in config.training.unsupervised else y_batch,
                        model.transform: np.concatenate((np.zeros([len(x_batch), 3]), adv_trans)) if 'semi' in config.training.unsupervised else adv_trans,
                        model.weights: [1. if i < len(x_batch) else LAMBDA for i in range((len(x_batch) + len(x_batch_adv)) if 'semi' in config.training.unsupervised else len(x_batch_adv))],
                        model.is_training: False}

        # Output to stdout
        if ii % num_output_steps == 0:
          nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
          adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
          print('Step {}:    ({})'.format(ii, datetime.now()))
          print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
          print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
          if ii != 0:
            print('    {} examples per second'.format(
                num_output_steps * batch_size / training_time))
            training_time = 0.0

        # Tensorboard summaries
        if ii % num_summary_steps == 0:
          summary = sess.run(adv_summaries, feed_dict=adv_dict)
          summary_writer.add_summary(summary, global_step.eval(sess))
          summary = sess.run(nat_summaries, feed_dict=nat_dict)
          summary_writer.add_summary(summary, global_step.eval(sess))

        # Write a checkpoint
        if ii % num_checkpoint_steps == 0:
          saver.save(sess,
                     os.path.join(model_dir, 'checkpoint'),
                     global_step=global_step)

        if eval_during_training and ii % num_eval_steps == 0:  
            attack.use_kl = False
            evaluate(model, attack, sess, config, eval_summary_writer)
            attack.use_kl = use_kl

        # Actual training step
        start = timer()
        if adversarial_training:
            adv_dict[model.is_training] = True
            sess.run(train_step, feed_dict=adv_dict)
        else:
            nat_dict[model.is_training] = True
            sess.run(train_step, feed_dict=nat_dict)
        end = timer()
        training_time += end - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Train script options',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str,
                        help='path to config file',
                        default='config.json', required=False)
    args = parser.parse_args()

    config_dict = utilities.get_config(args.config)

    model_dir = config_dict['model']['output_dir']
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    # keep the configuration file with the model for reproducibility
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, sort_keys=True, indent=4)

    config = utilities.config_to_namedtuple(config_dict)
    train(config)
