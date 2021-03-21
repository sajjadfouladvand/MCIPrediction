from __future__ import print_function

import os
import tensorflow as tf     
import random
import pdb
import numpy as np
from sklearn import metrics

num_time_steps =4
num_inputs = 39

class ReadingData(object):
    def __init__(self, max_seq_len=252, min_seq_len=3,
                  max_value=1000, path_t="", path_l="", path_s=""):
        
        self.data = []
        self.labels = []
        self.seqlen = []
        s=[]
        temp=[]
        #pdb.set_trace()
        with open(path_t) as f:
              for line in f:
                  d_temp=line.split(',')
                  for i in range(max_seq_len):
                      temp.append(float(d_temp[i]))
                      s.append(temp)
                      temp=[]
                  #pdb.set_trace()
                  dsds=s
                  self.data.append(s[1:])
                  s=[]
        #pdb.set_trace()
        d_temp=[]
        temp=[]
        with open(path_l) as f:
              for line in f:
                  d_temp=[]
                  d_temp=line.split(',')
                  temp.append(float(d_temp[0]))
                  temp.append(float(d_temp[1]))
                  temp.append(float(d_temp[2]))
                  self.labels.append(temp[1:])
                  temp=[]
        #pdb.set_trace()
        with open(path_s) as f:
              for line in f:
                  d_temp=[]
                  temp=[]
                  d_temp=line.split(',')
                  #pdb.set_trace()
                  self.seqlen.append(int(d_temp[1]))
        #pdb.set_trace()
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        if len(batch_data) < batch_size:
            batch_data = batch_data + (self.data[0:(batch_size - len(batch_data))])
            batch_labels = batch_labels + (self.labels[0:(batch_size - len(batch_labels))])
            batch_seqlen = batch_seqlen + (self.seqlen[0:(batch_size - len(batch_seqlen))])        
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen
def dynamicRNN(x, seqlen, weights, biases,seq_max_len,n_hidden, drop_out_keep_prob):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
   
    x = tf.unstack(x, num_time_steps, 1)
    
    # Define a lstm cell with tensorflow
    #lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    
    #========== Drop out layer
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,input_keep_prob=drop_out_keep_prob)#, output_keep_prob=keep_prob)    
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)
    #pdb.set_trace()
    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * 4 + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    softmax_predictions=tf.nn.softmax(tf.matmul(outputs, weights['out']) + biases['out'])
    #softmax_predictions=tf.nn.softmax(tf.matmul(outputs, weights['out']) + biases['out'])
    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out'], softmax_predictions


# ==========
#   MODEL
# ==========

# Parameters
def main(current_optimizer, current_threshold,cv_index, learning_rt,n_hid, batch_sz,num_itr, dropOutkeep,train_filename,train_labels_filename,train_lengths_filename,validation_filename,validation_labels_filename,validation_lengths_filename):
    tf.reset_default_graph() 
    learning_rate = learning_rt
    training_iters_up = num_itr#5000000 #1000000
    training_iters_low = 10000
    batch_size = batch_sz
    display_step = 150
    loss_threshold = 0.0001
    drop_out_keep_prob=dropOutkeep
# Network Parameters
    seq_max_len = 156 # Sequence max length
    n_hidden = n_hid # hidden layer num of features
    n_classes = 2 # linear sequence or not
    accuracies=[]
    print("=============================")
    print(train_filename)
    print(train_labels_filename)
    print(train_lengths_filename)
    print(validation_filename)
    print(validation_labels_filename)
    print(validation_lengths_filename)
    print("=============================")
    
    trainset = ReadingData(max_seq_len=(seq_max_len+1),path_t=train_filename, path_l=train_labels_filename, path_s=train_lengths_filename)
    testset = ReadingData(max_seq_len=(seq_max_len+1), path_t=validation_filename, path_l=validation_labels_filename,path_s=validation_lengths_filename)

# tf Graph input
    x = tf.placeholder("float", [None, num_time_steps, num_inputs])  # input sequence
    y = tf.placeholder("float", [None, n_classes])       # labels
# A placeholder for indicating each sequence length
    seqlen = tf.placeholder(tf.int32, [None])               # sequence length

# Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    pred, softmax_predictions = dynamicRNN(x, seqlen, weights, biases,seq_max_len,n_hidden, drop_out_keep_prob)

# Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    if current_optimizer==1:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    elif current_optimizer==2:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    pred_arg=tf.argmax(pred,1)
    y_arg=tf.argmax(y,1)
# Initializing the variables
    init = tf.global_variables_initializer()   
# Launch the graph
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    #pdb.set_trace()
    test_data_ar=np.array(test_data)
    test_data_ar_reshaped=np.reshape(test_data_ar,(len(test_data_ar), num_time_steps, num_inputs)).tolist()     
    test_seqlen_ar=np.array(test_seqlen)
    test_seqlen_ar_reshaped=(((test_seqlen_ar-1)/num_inputs).astype(int)+1).tolist()              
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #merged = tf.summary.merge_all() 
        #writer = tf.summary.FileWriter("logs", sess.graph) # Writing the summary
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        loss_file_name=os.path.join(os.path.dirname(__file__), 'Loss_Train'+str(cv_index)+'.csv')
        loss_test_file_name=os.path.join(os.path.dirname(__file__), 'Loss_Test_'+str(cv_index)+'.csv')
        with open (loss_file_name, 'w') as loss_file, open(loss_test_file_name, 'w') as loss_test_file:
            #print("Here is before entering trainig.")
            while step * batch_size < training_iters_up:
                #print("Here is after entering")
                batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
                batch_x_ar=np.array(batch_x)
                #batch_y_ar=np.array(batch_y)
                batch_seqlen_ar=np.array(batch_seqlen)
                batch_x_ar_reshaped=np.reshape(batch_x_ar,(batch_size, num_time_steps, num_inputs)).tolist()
                #batch_y_ar_reshaped=np.reshape(batch_y_ar,(batch_size, n_classes)).tolist()
                batch_seqlen_ar_reshaped=(((batch_seqlen_ar-1)/num_inputs).astype(int)+1).tolist()
                #batch_seq_ar_reshaped=np.reshape(batch_seqlen_ar,(batch_size, ))
                #print(np.shape(batch_x))
                #print(np.shape(batch_y))
                #print(np.shape(batch_seqlen))
                #print(type(batch_x))
                #print(type(batch_y))
                #print(type(batch_seqlen))
                #print(step)
                #if step==94:
                #   pdb.set_trace()
                # Run optimization op (backprop)
                #pdb.set_trace()
                #sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                #                        seqlen: batch_seqlen})
                sess.run(optimizer, feed_dict={x: batch_x_ar_reshaped, y: batch_y,
                                        seqlen: batch_seqlen_ar_reshaped})                
                #x_before_unstack, x_after_unstack=sess.run([x_before_unstack, x_after_unstack], feed_dict={x: batch_x, y: batch_y,
                #                        seqlen: batch_seqlen})                               
                if step % display_step == 0:
                    # Calculate batch accuracy
                    
                    #acc = sess.run(accuracy, feed_dict={x: batch_x_ar_reshaped, y: batch_y,
                    #                                    seqlen: batch_seqlen_ar_reshaped})
                    # Calculate batch loss
                    loss = sess.run(cost, feed_dict={x: batch_x_ar_reshaped, y: batch_y,
                                                    seqlen: batch_seqlen_ar_reshaped})
                    #THE PROBLEM IS HERE
                    loss_test = sess.run(cost, feed_dict={x: test_data_ar_reshaped, y: testset.labels,
                                                    seqlen: test_seqlen_ar_reshaped})                                                    
                    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                        "{:.6f}".format(loss))                                    
                    loss_file.write(str(loss))
                    loss_file.write(", ")
                    loss_test_file.write(str(loss_test))
                    loss_test_file.write(", ")                    
                    #if loss <= loss_threshold and step * batch_size > training_iters_low:
                    #    break
                    #result = sess.run(merged, feed_dict={x: batch_x, y: batch_y,
                    #                                    seqlen: batch_seqlen})
                    #writer.add_summary(result, step)  
                    #writer.add_graph(sess.graph)
                step += 1                
            #wait = input("PRESS ENTER TO CONTINUE.")
            #sys.exit()
        print("Optimization Finished!")
        # Calculate accuracy  
        accuracy_temp, y_arg_temp,pred_arg_temp, predictions_raw, softmax_predictions_temp =sess.run([accuracy,y_arg,pred_arg, pred, softmax_predictions], feed_dict={x: test_data_ar_reshaped, y: test_label,seqlen: test_seqlen_ar_reshaped})        
        #================= Using softmax of predictions for classofying
        #pdb.set_trace()
        probabilities_test=softmax_predictions_temp[:,0]
        #pdb.set_trace()
        if cv_index < 1000:
            current_threshold_temp=0
            thresholding_results=np.zeros((20, 10)) 
            for thresh_index in range(20):
                current_threshold_temp= current_threshold_temp + 0.05              
                tp=0
                tn=0
                fp=0
                fn=0
                for i in range(len(probabilities_test)):      
                    if(probabilities_test[i]<current_threshold_temp and y_arg_temp[i]==1):
                        tn=tn+1
                    elif(probabilities_test[i]>=current_threshold_temp and y_arg_temp[i]==0):
                        tp=tp+1
                    elif(probabilities_test[i]>=current_threshold_temp and y_arg_temp[i]==1):
                        fp=fp+1
                    elif(probabilities_test[i]<current_threshold_temp and y_arg_temp[i]==0):
                        fn=fn+1
                if((tp+fp)==0):
                    precision_thresholding=0
                else:
                    precision_thresholding=tp/(tp+fp)
                recall_thresholding=tp/(tp+fn)
                sensitivity_thresholding=tp/(tp+fn)
                specificity_thresholding=tn/(tn+fp)    
                if (precision_thresholding+recall_thresholding) !=0:
                    F1Score_thresholding=(2*precision_thresholding*recall_thresholding)/(precision_thresholding+recall_thresholding)      
                else:
                    F1Score_thresholding=0        
                accuracy_thresholding= (tp+tn)/(tp+tn+fp+fn)
                thresholding_results[thresh_index, 0] = tp
                thresholding_results[thresh_index, 1] = tn
                thresholding_results[thresh_index, 2] = fp
                thresholding_results[thresh_index, 3] = fn
                thresholding_results[thresh_index, 4] = accuracy_thresholding
                thresholding_results[thresh_index, 5] = specificity_thresholding
                thresholding_results[thresh_index, 6] = precision_thresholding
                thresholding_results[thresh_index, 7] = recall_thresholding
                thresholding_results[thresh_index, 8] = F1Score_thresholding
                thresholding_results[thresh_index, 9] = current_threshold_temp
            #pdb.set_trace()
            best_validation_results=thresholding_results[np.argmax(thresholding_results[:,8]),:]
            tp=best_validation_results[0]
            tn=best_validation_results[1]
            fp=best_validation_results[2]
            fn=best_validation_results[3]
            accuracy_thresholding=best_validation_results[4]
            specificity_thresholding=best_validation_results[5]
            precision_thresholding=best_validation_results[6]
            recall_thresholding=best_validation_results[7]
            F1Score_thresholding=best_validation_results[8]
            optimum_threshold=best_validation_results[9]            
        elif cv_index == 1000:
            optimum_threshold= current_threshold              
            tp=0
            tn=0
            fp=0
            fn=0
            for i in range(len(probabilities_test)):      
                if(probabilities_test[i]<optimum_threshold and y_arg_temp[i]==1):
                    tn=tn+1
                elif(probabilities_test[i]>=optimum_threshold and y_arg_temp[i]==0):
                    tp=tp+1
                elif(probabilities_test[i]>=optimum_threshold and y_arg_temp[i]==1):
                    fp=fp+1
                elif(probabilities_test[i]<optimum_threshold and y_arg_temp[i]==0):
                    fn=fn+1
            #pdb.set_trace()
            if( (tp+fp)==0):
                precision_thresholding=0
            else:
                precision_thresholding=tp/(tp+fp)
            recall_thresholding=tp/(tp+fn)
            sensitivity_thresholding=tp/(tp+fn)
            specificity_thresholding=tn/(tn+fp)    
            if (precision_thresholding+recall_thresholding) !=0:
                F1Score_thresholding=(2*precision_thresholding*recall_thresholding)/(precision_thresholding+recall_thresholding)      
            else:
                F1Score_thresholding=0        
            accuracy_thresholding= (tp+tn)/(tp+tn+fp+fn)        
            #pdb.set_trace()
        print("=================== Testing F1 score using thresholding method: ", F1Score_thresholding)
        np.savetxt("softmax_predictions"+str(cv_index)+".csv",softmax_predictions_temp, delimiter=',')
        y_arg_temp_flipped=[0 if x==1 else 1 for x in y_arg_temp]
        probabilities_posClass=softmax_predictions_temp[:,0]
        probabilities_negClass=softmax_predictions_temp[:,1]
        fpr_pos, tpr_pos, thresholds_pos = metrics.roc_curve(y_arg_temp_flipped, probabilities_posClass, pos_label=1)
        pos_auc=metrics.auc(fpr_pos, tpr_pos)
        print("================= Testing AUC:", pos_auc)
        fpr_negs, tpr_negs, thresholds_negs = metrics.roc_curve(y_arg_temp_flipped, probabilities_negClass, pos_label=0)
        neg_auc=metrics.auc(fpr_negs, tpr_negs)        
        #print(predictions_raw)
        #pdb.set_trace()
        train_data = trainset.data
        train_label = trainset.labels
        train_seqlen = trainset.seqlen
        train_data_ar = np.array(train_data)
        train_data_ar_reshaped=np.reshape(train_data_ar,(len(train_data_ar), num_time_steps, num_inputs)).tolist()     
        train_seqlen_ar=np.array(train_seqlen)
        train_seqlen_ar_reshaped=(((train_seqlen_ar-1)/num_inputs).astype(int)+1).tolist()                                
        accuracy_temp_train, y_arg_temp_train,pred_arg_temp_train, softmax_predictions_train =sess.run([accuracy,y_arg,pred_arg, softmax_predictions], feed_dict={x: train_data_ar_reshaped, y: train_label,seqlen: train_seqlen_ar_reshaped})

        probabilities_train=softmax_predictions_train[:,0]
        current_threshold_train_temp=0
        thresholding_results_train=np.zeros((20, 10)) 
        #pdb.set_trace()
        for thresh_index in range(20):
            current_threshold_train_temp= current_threshold_train_temp + 0.05              
            tp_train=0
            tn_train=0
            fp_train=0
            fn_train=0
            for i in range(len(probabilities_train)):      
                if(probabilities_train[i]<current_threshold_train_temp and y_arg_temp_train[i]==1):
                    tn_train=tn_train+1
                elif(probabilities_train[i]>=current_threshold_train_temp and y_arg_temp_train[i]==0):
                    tp_train=tp_train+1
                elif(probabilities_train[i]>=current_threshold_train_temp and y_arg_temp_train[i]==1):
                    fp_train=fp_train+1
                elif(probabilities_train[i]<current_threshold_train_temp and y_arg_temp_train[i]==0):
                    fn_train=fn_train+1
            if( (tp_train+fp_train)==0):
                precision_thresholding_train=0
            else:
                precision_thresholding_train=tp_train/(tp_train+fp_train)
            recall_thresholding_train=tp_train/(tp_train+fn_train)
            sensitivity_thresholding_train=tp_train/(tp_train+fn_train)
            specificity_thresholding_train=tn_train/(tn_train+fp_train)    
            if (precision_thresholding_train+recall_thresholding_train) !=0:
                F1Score_thresholding_train=(2*precision_thresholding_train*recall_thresholding_train)/(precision_thresholding_train+recall_thresholding_train)      
            else:
                F1Score_thresholding_train=0        
            accuracy_thresholding_train= (tp_train+tn_train)/(tp_train+tn_train+fp_train+fn_train)
            thresholding_results_train[thresh_index, 0] = tp_train
            thresholding_results_train[thresh_index, 1] = tn_train
            thresholding_results_train[thresh_index, 2] = fp_train
            thresholding_results_train[thresh_index, 3] = fn_train
            thresholding_results_train[thresh_index, 4] = accuracy_thresholding_train
            thresholding_results_train[thresh_index, 5] = specificity_thresholding_train
            thresholding_results_train[thresh_index, 6] = precision_thresholding_train
            thresholding_results_train[thresh_index, 7] = recall_thresholding_train
            thresholding_results_train[thresh_index, 8] = F1Score_thresholding_train
            thresholding_results_train[thresh_index, 9] = current_threshold_train_temp
        #pdb.set_trace()
        best_validation_results_train=thresholding_results_train[np.argmax(thresholding_results_train[:,8]),:]
        tp_train=best_validation_results_train[0]
        tn_train=best_validation_results_train[1]
        fp_train=best_validation_results_train[2]
        fn_train=best_validation_results_train[3]
        accuracy_thresholding_train=best_validation_results_train[4]
        specificity_thresholding_train=best_validation_results_train[5]
        precision_thresholding_train=best_validation_results_train[6]
        recall_thresholding_train=best_validation_results_train[7]
        F1Score_thresholding_train=best_validation_results_train[8]
        optimum_threshold_train=best_validation_results_train[9]
        print("=================== Training F1 score using thresholding method: ", F1Score_thresholding_train)
        #pdb.set_trace()        
        y_arg_temp_train_flipped=[0 if x==1 else 1 for x in y_arg_temp_train]
        probabilities_posClass_train=softmax_predictions_train[:,0]
        probabilities_negClass_train=softmax_predictions_train[:,1]
        fpr_pos_train, tpr_pos_train, thresholds_pos_train = metrics.roc_curve(y_arg_temp_train_flipped, probabilities_posClass_train, pos_label=1)
        pos_auc_train=metrics.auc(fpr_pos_train, tpr_pos_train)       
        fpr_negs_train, tpr_negs_train, thresholds_negs_train = metrics.roc_curve(y_arg_temp_train_flipped, probabilities_negClass_train, pos_label=0)
        neg_auc_train=metrics.auc(fpr_negs_train, tpr_negs_train)      
    return optimum_threshold, pos_auc, neg_auc, pos_auc_train, neg_auc_train, accuracy_thresholding, precision_thresholding, recall_thresholding, F1Score_thresholding, specificity_thresholding, tp, tn, fp, fn, accuracy_thresholding_train, precision_thresholding_train, recall_thresholding_train, F1Score_thresholding_train, specificity_thresholding_train,tp_train, tn_train, fp_train, fn_train

if __name__ == "__main__": main(current_optimizer, current_threshold, cv_index, learning_rt,n_hid, batch_sz,num_itr, dropOutkeep, train_filename,train_labels_filename,train_lengths_filename,validation_filename,validation_labels_filename,validation_lengths_filename)