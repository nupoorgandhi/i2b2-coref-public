#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf
import util
import logging
import sys
from tensorflow.python import debug as tf_debug
import numpy as np
import copy
format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

if __name__ == "__main__":
  print('ented main function', os.environ)


  config1 = util.initialize_from_env()
  projection_args = util.initialize_yaml_from_env(config1['projection_yaml'])

  report_frequency = config1["report_frequency"]
  eval_frequency = config1["eval_frequency"]

  model = util.get_model(config1, projection_args)
  saver = tf.train.Saver()

  log_dir = config1["log_dir"]
  max_steps = config1['num_epochs'] * config1['num_docs']
  if len(str(config1['sample_id'])) > 1 :
    print('list of elemenrs', str(config1['sample_id']).split('-')[0])

    num_target = int(str(config1['sample_id']).split('-')[0])
    print('num_target', num_target)
    max_steps += config1['num_epochs'] * num_target
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)
  print('max_steps', max_steps)

  model_name = sys.argv[1]
  max_f1 = 0
  prev_f1 = -1
  prev_f1_worse = False
  mode = 'w'


  # try to solve GPu issue
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.4

  # config.gpu_options.allow_growth = True
  with tf.Session(config=config) as session:
    # session = tf_debug.LocalCLIDebugWrapperSession(session)

    # print('(just started session) gpu memory used',session.run(tf.contrib.memory_stats.MaxBytesInUse()))



  # with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    # print('(just initialized variables)gpu memory used', session.run(tf.contrib.memory_stats.MaxBytesInUse()))

    model.start_enqueue_thread(session)
    # print('(just started thread)gpu memory used', session.run(tf.contrib.memory_stats.MaxBytesInUse()))

    accumulated_loss = 0.0
    print('log_dir', log_dir)
    ckpt = tf.train.get_checkpoint_state(log_dir)
    print('ckpt:', ckpt)
    if ckpt and ckpt.model_checkpoint_path:
      print("Restoring from: {}".format(ckpt.model_checkpoint_path))
      saver.restore(session, ckpt.model_checkpoint_path)
      mode = 'a'
      print('restored model')
    else:
      print('did not find model, this is good')
    print('(just loaded model)gpu memory used')




    fh = logging.FileHandler(os.path.join(log_dir, 'stdout.log'), mode=mode)
    fh.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)

    initial_time = time.time()
    accumulated_to_print = []
    while True:

      # print('starting model session')
      # print('(about to run session)gpu memory used', session.run(tf.contrib.memory_stats.MaxBytesInUse()))

      tf_loss_both, tf_global_step, _  = session.run([model.loss, model.global_step, model.train_op])
      to_print = tf_loss_both[1]
      tf_loss = tf_loss_both[0]


      accumulated_loss += tf_loss
      accumulated_to_print.append(to_print)

      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        print('things', tf_global_step, average_loss, steps_per_second)
        logger.info("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
        writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0
        accumulated_to_print = np.array(accumulated_to_print)
        print('to_print',np.mean(accumulated_to_print))
        accumulated_to_print = []


      if tf_global_step  > 0 and tf_global_step % eval_frequency == 0:
        saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
        eval_summary_dict, eval_f1 = model.evaluate(session,sys.argv[1], tf_global_step)
        eval_summary_dict_copy = copy.deepcopy(eval_summary_dict)
        eval_summary_dict_copy.pop('projection_loss', None)
        eval_summary = util.make_summary(eval_summary_dict_copy)
        # print('[main]returned from evaluation')

        if eval_f1 > max_f1:
          print('[main]eval_f1 > max_f1')
          max_f1 = eval_f1
          util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

        writer.add_summary(eval_summary, tf_global_step)
        # print('[main] added summary')
        writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)
        if 'projection_loss' in eval_summary_dict:
          logger.info("[{}] evaL_f1={:.4f}, max_f1={:.4f}, proj_loss={}".format(tf_global_step, eval_f1, max_f1, eval_summary_dict['projection_loss']))
        else:
          logger.info("[{}] evaL_f1={:.4f}, max_f1={:.4f}".format(tf_global_step, eval_f1, max_f1))
        # print('[main] added log')



        saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
        print('[main] finished saving')
        print('[main] tf global_step', tf_global_step, 'max_steps', max_steps)


        if tf_global_step > 61500:
          break
        if eval_f1 < prev_f1 and prev_f1_worse and 'tuneTrue' in model_name:# and len(str(config1['sample_id'])) > 1 :
            print('previous f1 was worse')
            break
        elif eval_f1 < prev_f1:
            prev_f1_worse = True
            prev_f1 = eval_f1
        else:
            prev_f1_worse = False
            prev_f1 = eval_f1

        if (tf_global_step > max_steps or tf_global_step > 60200) and  len(str(config1['sample_id'])) <= 1 and 'tuneTrue' not in model_name:
          print('[main] exceeded max_steps', 'config[num_epochs]', config1['num_epochs'], 'config[num_docs]', config1['num_docs'], 'tf_global_step', tf_global_step)
          break