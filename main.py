from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os
import importlib
import numpy as np
import tensorflow as tf
import texar.tf as tx

from ctrl_gen_model import CtrlGenModel

DATA_DIR = 'rt_data/'



flags = tf.flags

flags.DEFINE_string('config', 'config', 'The config to use.')

FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)
SEED = 123
os.environ['PYTHONHASHSEED']=str(0)
random.seed(SEED)
np.random.seed(SEED)
rs = np.random.RandomState(123)

try:
    tf.random.set_random_seed(123)
except:
    tf.set_random_seed(123)


def _main(_):
    # Data
    train_autoencoder = tx.data.MultiAlignedData(config.train_autoencoder)
    dev_autoencoder = tx.data.MultiAlignedData(config.dev_autoencoder)
    test_autoencoder = tx.data.MultiAlignedData(config.test_autoencoder)
    train_discriminator = tx.data.MultiAlignedData(config.train_discriminator)
    dev_discriminator = tx.data.MultiAlignedData(config.dev_discriminator)
    test_discriminator = tx.data.MultiAlignedData(config.test_discriminator)
    train_defender = tx.data.MultiAlignedData(config.train_defender)
    test_defender = tx.data.MultiAlignedData(config.test_defender)
    vocab = train_discriminator.vocab(0)

    iterator = tx.data.FeedableDataIterator(
        {
            'train_autoencoder': train_autoencoder,
            'dev_autoencder': dev_autoencoder,
            'test_autoencoder': test_autoencoder,
            'train_discriminator': train_discriminator,
            'dev_discriminator': dev_discriminator,
            'test_discriminator': test_discriminator,
            'train_defender': train_defender,
            'test_defender': test_defender,
        })
    batch = iterator.get_next()

    # Model
    gamma = tf.placeholder(dtype=tf.float32, shape=[], name='gamma')
    lambda_D = tf.placeholder(dtype=tf.float32, shape=[], name='lambda_g')
    lambda_ae_= 1.0
    model = CtrlGenModel(batch, vocab, lambda_ae_, gamma, lambda_D, config.model)

    def autoencoder(sess,lambda_ae_, gamma_, lambda_D_, epoch, mode, verbose=True):
        avg_meters_g = tx.utils.AverageRecorder(size=10)
        step = 0
        if mode == "train":
            dataset = "train_autoencoder"
            while True:
                try:
                    step += 1
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, dataset),
                        gamma: gamma_,
                        lambda_D: lambda_D_,
                    }
                    vals_g = sess.run(model.fetches_train_g, feed_dict=feed_dict)
                    loss_g_ae_summary = vals_g.pop("loss_g_ae_summary")
                    loss_g_clas_summary = vals_g.pop("loss_g_clas_summary")
                    avg_meters_g.add(vals_g)

                    if verbose and (step == 1 or step % config.display == 0):
                        print('step: {}, {}'.format(step, avg_meters_g.to_str(4)))

                    if verbose and step % config.display_eval == 0:
                        iterator.restart_dataset(sess, 'dev_autoencoder')
                        _eval_epoch(sess,lambda_ae_, gamma_, lambda_ae_, epoch)

                except tf.errors.OutOfRangeError:
                    print('epoch: {}, {}'.format(epoch, avg_meters_g.to_str(4)))
                    break
        else:
            dataset = "test_autoencoder"
            while True:
                try:
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, dataset),
                        gamma: gamma_,
                        lambda_D: lambda_D_,
                        tx.context.global_mode(): tf.estimator.ModeKeys.EVAL
                    }

                    vals = sess.run(model.fetches_eval, feed_dict=feed_dict)

                    samples = tx.utils.dict_pop(vals, list(model.samples.keys()))
                    hyps = tx.utils.map_ids_to_strs(samples['transferred'], vocab)
                    refs = tx.utils.map_ids_to_strs(samples['original'], vocab)
                    refs = np.expand_dims(refs, axis=1)
                    avg_meters_g.add(vals)
                    # Writes samples
                    tx.utils.write_paired_text(
                        refs.squeeze(), hyps,
                        os.path.join(config.sample_path, 'val.%d' % epoch),
                        append=True, mode='v')

                except tf.errors.OutOfRangeError:
                    print('{}: {}'.format(
                        "test_autoencoder_only", avg_meters_g.to_str(precision=4)))
                    break
    def discriminator(sess,lambda_ae_, gamma_, lambda_D_, epoch, mode, verbose=True):
        avg_meters_d = tx.utils.AverageRecorder(size=10)
        y_true = []
        y_pred = []
        y_prob=[]
        sentences = []
        step = 0
        if mode == "train":
            dataset = "train_discriminator"
            
            while True:
                try:
                    step += 1
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, dataset),
                        gamma: gamma_,
                        lambda_D: lambda_D_,
                    }

                    vals_d = sess.run(model.fetches_train_d, feed_dict=feed_dict)
                    y_pred.extend(vals_d.pop("y_pred").tolist())
                    y_true.extend(vals_d.pop("y_true").tolist())
                    y_prob.extend(vals_d.pop("y_prob").tolist())
                    sentences.extend(vals_d.pop("sentences").tolist())
                    avg_meters_d.add(vals_d)

                    # if verbose and (step == 1 or step % config.display == 0):
                    if verbose and step%40==0 :
                        print('step: {}, {}'.format(step, avg_meters_d.to_str(4)))                

                except tf.errors.OutOfRangeError:                    
                    iterator.restart_dataset(sess, 'dev_discriminator')
                    _,_,_,_,val_acc=_eval_discriminator(sess,lambda_ae_, gamma_, lambda_D_, epoch,'dev_discriminator')
                    return val_acc

        if mode == 'test' :
            dataset = "test_discriminator"
            iterator.restart_dataset(sess, dataset)
            y_pred,y_true,y_prob,sentences,_=_eval_discriminator(sess,
                                            lambda_ae_, gamma_, lambda_D_, epoch, dataset)                    

            assert(len(y_pred)==len(y_true)==len(y_prob)==len(sentences))
            
            # tp=0
            # tn=0
            # fp=0
            # acc=0
            # for sent,label,pred,prob in zip(sentences,y_true,y_pred,y_prob):
            #     if pred==1 and label==1:
            #         tp+=1.0/len(y_true)
            #     if pred==0 and label==0:
            #         tn+=1.0/len(y_true)                    
            #     if pred==1 and label==0:
            #         fp+=1.0/len(y_true)                                        
            #     if pred==label:
            #         acc+=1.0/len(y_true)

            # print('true_positives:{}'.format(tp))
            # print('true_negatives:{}'.format(tn))
            # print('false_positives:{}'.format(fp))
            # print('accuracy:{}'.format(acc))
                
            # with open('prob_vocab.txt', 'w') as file:
            #     for word, prob_values in zip(sentences,y_prob):
            #         file.write(word)                        
            #         file.write('\t')
            #         file.write(str(prob_values))
            #         file.write('\n')            

            # txt=open('rand_sent_from_vocab_Discriminator_label.txt','w')
            # with open('rand_sent_from_vocab_Discriminator.txt', 'w') as file:
            #     for sentence, pred in zip(sentences, y_pred):
            #             file.write(sentence+'\n')
            #             txt.write(str(pred)+'\n')
            txt = open(DATA_DIR+'rand_x_sent_from_vocab_Discriminator_neg_confirmed.txt', 'w')
            with open(DATA_DIR+'rand_x_sent_from_vocab_Discriminator_neg_confirmed.txt', 'w') as file:
                for sentence, pred,label in zip(sentences, y_pred,y_true):
                    if pred==0 and label==0:
                        file.write(sentence+'\n')
                        txt.write(str(pred)+'\n')            
                        
    def defender(sess,lambda_ae_, gamma_, lambda_D_, epoch, mode, verbose=True):
        avg_meters_g = tx.utils.AverageRecorder(size=10)
        step = 0
        if mode == "train":
            dataset = "train_defender"
            while True:
                try:
                    step += 1
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, dataset),
                        gamma: gamma_,
                        lambda_D: lambda_D_,
                    }
                    vals_g = sess.run(model.fetches_train_g, feed_dict=feed_dict)
                    loss_g_ae_summary = vals_g.pop("loss_g_ae_summary")
                    loss_g_clas_summary = vals_g.pop("loss_g_clas_summary")
                    avg_meters_g.add(vals_g)

                    if verbose and (step == 1 or step % config.display == 0):
                        print('step: {}, {}'.format(step, avg_meters_g.to_str(4)))

                except tf.errors.OutOfRangeError:
                    print('epoch: {}, {}'.format(epoch, avg_meters_g.to_str(4)))
                    break
        else:
            dataset = "test_defender"
            while True:
                try:
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, dataset),
                        gamma: gamma_,
                        lambda_D: lambda_D_,
                        tx.context.global_mode(): tf.estimator.ModeKeys.EVAL
                    }

                    vals = sess.run(model.fetches_eval, feed_dict=feed_dict)

                    samples = tx.utils.dict_pop(vals, list(model.samples.keys()))
                    hyps = tx.utils.map_ids_to_strs(samples['transferred'], vocab)
                    refs = tx.utils.map_ids_to_strs(samples['original'], vocab)
                    refs = np.expand_dims(refs, axis=1)
                    avg_meters_g.add(vals)
                    # Writes samples
                    tx.utils.write_paired_text(
                        refs.squeeze(), hyps,
                        os.path.join(config.sample_path, 'defender_val.%d' % epoch),
                        append=True, mode='v')

                except tf.errors.OutOfRangeError:
                    print('{}: {}'.format(
                        "test_defender", avg_meters_g.to_str(precision=4)))
                    break                
    def _eval_discriminator(sess,lambda_ae_, gamma_, lambda_D_, epoch, dataset):
        avg_meters_d = tx.utils.AverageRecorder()
        y_true = []
        y_pred = []
        y_prob=[]
        sentences = []
        while True:
            try:
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, dataset),
                    gamma: gamma_,
                    lambda_D: lambda_D_,
                }

                vals_d = sess.run(model.fetches_dev_test_d, feed_dict=feed_dict)
                y_pred.extend(vals_d.pop("y_pred").tolist())
                y_true.extend(vals_d.pop("y_true").tolist())
                y_prob.extend(vals_d.pop("y_prob").tolist())
                sentence=vals_d.pop("sentences").tolist()
                sentences.extend(tx.utils.map_ids_to_strs(sentence, vocab))
                batch_size = vals_d.pop('batch_size')
                avg_meters_d.add(vals_d, weight=batch_size)                                                            

            except tf.errors.OutOfRangeError:
                acc = avg_meters_d.avg()['accu_d']
                print('{}: {}'.format(dataset, avg_meters_d.to_str(precision=4)))
                break

        return y_pred,y_true,y_prob,sentences, acc  
    tf.gfile.MakeDirs(config.sample_path)
    tf.gfile.MakeDirs(config.checkpoint_path)

    # Runs the logics
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        saver = tf.train.Saver(max_to_keep=None)
        print(config.restore)
        if config.restore:
            print('Restore from: {}'.format(config.restore))
            saver.restore(sess, config.restore)

        iterator.initialize_dataset(sess)
        
        gamma_ = 1.0
        lambda_D_ = 0.0
        

        prev_acc = 0
        # #Train discriminator.
        for epoch in range(1, config.discriminator_nepochs + 1):
            print("Epoch number:", epoch)
            iterator.restart_dataset(sess, ['train_discriminator'])
            val_acc = discriminator(sess,lambda_ae_, gamma_, lambda_D_, epoch, mode='train')
            if(val_acc > prev_acc):
                print("Accuracy is better, saving model")   
                prev_acc = val_acc         
                saver.save(
                    sess, os.path.join(config.checkpoint_path, 'discriminator_only_ckpt'), epoch)
            else:
                print("Accuracy is worse")
        # Test discriminator.
        iterator.restart_dataset(sess, ['test_discriminator'])
        print('gamma:{}'.format(gamma_))
        discriminator(sess,lambda_ae_, gamma_, lambda_D_, 1, mode='test')

        exit()
        
        # Train autoencoder
        for epoch in range(1, config.autoencoder_nepochs + 1):
            iterator.restart_dataset(sess, ['train_autoencoder'])
            autoencoder(sess,lambda_ae_, gamma_, lambda_D_, epoch, mode='train')
            saver.save(
                sess, os.path.join(config.checkpoint_path, 'discriminator_only_and_autoencoder_only_ckpt'), epoch)
        

        # Test autoencoder
        iterator.restart_dataset(sess, ['test_autoencoder'])
        autoencoder(sess,lambda_ae_, gamma_, lambda_D_, 1, mode='test')


        gamma_ = 1.0    
        lambda_D_ = 1.0
        # # gamma_decay = 0.5  # Gumbel-softmax temperature anneal rate

                # Train Defender
        for epoch in range(0, config.full_nepochs ):
            # gamma_ = max(0.001, gamma_ * 0.5)
            print('gamma: {}, lambda_ae: {}, lambda_D: {}'.format(gamma_,lambda_ae_, lambda_D_))
        
            iterator.restart_dataset(sess, ['train_defender'])
            defender(sess,lambda_ae_, gamma_, lambda_D_, epoch, mode='train')
            saver.save(sess, os.path.join(config.checkpoint_path, 'full_ckpt'), epoch)

       	        # Test Defender
        iterator.restart_dataset(sess, 'test_defender')
        defender(sess,lambda_ae_, gamma_, lambda_D_, 1, mode='test')


if __name__ == '__main__':
    tf.app.run(main=_main)





















