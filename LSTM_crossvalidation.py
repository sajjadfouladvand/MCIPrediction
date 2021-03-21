import  LSTM_model_updated as lsad
import os
import random 
import pdb
import numpy as np
patient_sequences=[]
labels=[]
lengths=[]
#===== Parameter pools
LR_pool=[0.1]#[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.08, 0.1]
num_hidden_pool=[140]#[10, 20, 40, 60, 80, 100, 140, 180, 200, 300, 600]
batch_size_pool=[64]#[32, 64, 128, 256, 512]
dropout_prob_pool=[0.4]#[0.3, 0.4, 0.5, 0.6, 0.7]
num_iterations_pool=[10000]#[100000]#[10000, 100000, 1000000, 2000000, 5000000, 10000000, 15000000]
thresholds_pool=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
optimizers_pool=[1]#[1, 2] # 1 means GDO and 2 means Adam
num_combinations=3
cv_results=np.zeros((num_combinations, 25), dtype=np.float64)
#==============================
train_filename='patient_data_included_noMissing_training_oversampled.csv'
train_labels_filename='patient_data_included_noMissing_labels_training_oversampled.csv'
train_lengths_filename='patient_data_included_noMissing_lengths_training_oversampled.csv'
validation_filename='patient_data_included_noMissing_validation.csv'
validation_labels_filename='patient_data_included_noMissing_labels_validation.csv'
validation_lengths_filename='patient_data_included_noMissing_lengths_validation.csv'

trainANDval_filename='patient_data_included_noMissing_trainANDvalidation_oversampled.csv'
trainANDval_labels_filename='patient_data_included_noMissing_labels_trainANDvalidation_oversampled.csv'
trainANDval_lengths_filename='patient_data_included_noMissing_lengths_trainANDvalidation_oversampled.csv'

testing_filename='patient_data_included_noMissing_testing.csv'
testing_labels_filename='patient_data_included_noMissing_labels_testing.csv'
testing_lengths_filename='patient_data_included_noMissing_lengths_testing.csv'

res_fn=os.path.join(os.path.dirname(__file__), 'Results_lstm.csv')
header_results_filename= "Experiment, Learning Rate, Number of Hidden Neurons, Batch Size, Num Iterations, Drop Out Prob, Optimizer (1 means GDO and 2 means Adam), Best validation threshold ,accuracy, precision, recall, F1 Score, specificity, AUC for positive class, AUC for negative class ,tp, tn, fp, fn, accuracy_train, precision_train, recall_train, F1 Score_train, specificity_train, AUC for positive class, AUC for negative class, tp_train, tn_train, fp_train, fn_train"
with open(res_fn, 'w') as results_file:
    results_file.write("".join(["".join(x) for x in header_results_filename]))  
    results_file.write("\n")    
    #pdb.set_trace()
    for i in range(num_combinations):
        print(i)
        results_file.write("Fold "+ str(i))
        results_file.write(",")
        current_LR=random.choice(LR_pool)
        cv_results[i, 0]=current_LR
        results_file.write(str(current_LR))
        results_file.write(",")
        current_nHidden=random.choice(num_hidden_pool)
        cv_results[i,1]=current_nHidden
        results_file.write(str(current_nHidden))
        results_file.write(",")
        current_batchS=random.choice(batch_size_pool)
        cv_results[i,2]=current_batchS
        results_file.write(str(current_batchS))
        results_file.write(",")
        current_numItr=random.choice(num_iterations_pool)
        cv_results[i,3]=current_numItr
        results_file.write(str(current_numItr))
        results_file.write(",")
        current_dropOut=random.choice(dropout_prob_pool)
        cv_results[i,4]=current_dropOut
        results_file.write(str(current_dropOut))
        results_file.write(",")
        current_optimizer=random.choice(optimizers_pool)
        cv_results[i,5]=current_optimizer
        results_file.write(str(current_optimizer))
        results_file.write(",")        
        current_threshold=random.choice(thresholds_pool)
        #pdb.set_trace()
        best_threshold, pos_auc, neg_auc, pos_auc_train, neg_auc_train, accuracy_temp, precision, recall, F1Score, specificity, tp, tn, fp, fn, accuracy_temp_train, precision_train, recall_train, FScore_train, specificity_train,tp_train, tn_train, fp_train, fn_train=lsad.main(current_optimizer, current_threshold, i, current_LR,current_nHidden, current_batchS,current_numItr, current_dropOut, train_filename,train_labels_filename,train_lengths_filename,validation_filename,validation_labels_filename,validation_lengths_filename)
        cv_results[i,6]=best_threshold
        results_file.write(str(best_threshold))
        results_file.write(",")        
        results_file.write(str(accuracy_temp))
        cv_results[i, 7]=accuracy_temp
        results_file.write(", ")
        results_file.write(str(precision))
        cv_results[i, 8]=precision
        results_file.write(", ")
        results_file.write(str(recall))
        cv_results[i, 9]=recall
        results_file.write(", ")
        results_file.write(str(F1Score))
        cv_results[i, 10]=F1Score
        results_file.write(", ")
        results_file.write(str(specificity))
        cv_results[i, 11]=specificity
        results_file.write(", ")
        results_file.write(str(pos_auc))
        results_file.write(", ")     
        results_file.write(str(neg_auc))
        results_file.write(", ")                     
        results_file.write(str(tp))
        cv_results[i, 12]=tp
        results_file.write(", ")
        results_file.write(str(tn))
        cv_results[i, 13]=tn
        results_file.write(", ")
        results_file.write(str(fp))
        cv_results[i, 14]=fp
        results_file.write(", ")
        results_file.write(str(fn))
        cv_results[i, 15]=fn
        results_file.write(", ")
        results_file.write(str(accuracy_temp_train))
        cv_results[i, 16]=accuracy_temp_train
        results_file.write(", ")
        results_file.write(str(precision_train))
        cv_results[i, 17]=precision_train
        results_file.write(", ")
        results_file.write(str(recall_train))
        cv_results[i, 18]=recall_train
        results_file.write(", ")
        results_file.write(str(FScore_train))
        cv_results[i, 19]=FScore_train
        results_file.write(", ")
        results_file.write(str(specificity_train))
        cv_results[i, 20]=specificity_train
        results_file.write(", ")
        results_file.write(str(pos_auc_train))
        results_file.write(", ")            
        results_file.write(str(neg_auc_train))
        results_file.write(", ")                     
        results_file.write(str(tp_train))
        cv_results[i, 21]=tp_train
        results_file.write(", ")
        results_file.write(str(tn_train))
        cv_results[i, 22]=tn_train
        results_file.write(", ")
        results_file.write(str(fp_train))
        cv_results[i, 23]=fp_train
        results_file.write(", ")
        results_file.write(str(fn_train)) 
        results_file.write("\n")        
        cv_results[i, 24]=fn_train       
    #pdb.set_trace()
    best_cv_result=cv_results[np.argmax(cv_results[:,10]),:]
    optimizers_best= cv_results[i, 5]
    threshold_best= cv_results[i, 6]
    cv_index = 1000
    learning_rate_best = cv_results[i, 0]
    num_Hidden_best = cv_results[i, 1]
    batch_size_best = cv_results[i, 2]
    num_iteration_best = cv_results[i, 3]
    dropout_best = cv_results[i, 4]    
    #accuracy_temp, precision, recall, F1Score, specificity, tp, tn, fp, fn, accuracy_temp_train, precision_train, recall_train, FScore_train, specificity_train,tp_train, tn_train, fp_train, fn_train=lsad.main(best_cv_result[0],best_cv_result[1], best_cv_result[2],best_cv_result[3], best_cv_result[4], trainANDval_filename,trainANDval_labels_filename,trainANDval_lengths_filename,testing_filename,testing_labels_filename,testing_lengths_filename)        
    #pdb.set_trace()
    best_threshold, pos_auc, neg_auc, pos_auc_train, neg_auc_train, accuracy_temp, precision, recall, F1Score, specificity, tp, tn, fp, fn, accuracy_temp_train, precision_train, recall_train, FScore_train, specificity_train,tp_train, tn_train, fp_train, fn_train=lsad.main(optimizers_best, threshold_best, cv_index, learning_rate_best, int(num_Hidden_best), int(batch_size_best),int(num_iteration_best), dropout_best, trainANDval_filename, trainANDval_labels_filename, trainANDval_lengths_filename,testing_filename,testing_labels_filename,testing_lengths_filename)    
    results_file.write("Testing")
    results_file.write(",")
    results_file.write(str(learning_rate_best))
    results_file.write(", ")
    results_file.write(str(num_Hidden_best))
    results_file.write(", ")
    results_file.write(str(batch_size_best))
    results_file.write(", ")
    results_file.write(str(num_iteration_best))
    results_file.write(", ")
    results_file.write(str(dropout_best))
    results_file.write(", ") 
    results_file.write(str(optimizers_best))
    results_file.write(", ") 
    results_file.write(str(threshold_best))
    results_file.write(", ")     
    results_file.write(str(accuracy_temp))
    results_file.write(", ")
    results_file.write(str(precision))
    results_file.write(", ")
    results_file.write(str(recall))
    results_file.write(", ")
    results_file.write(str(F1Score))
    results_file.write(", ")
    results_file.write(str(specificity))
    results_file.write(", ")
    results_file.write(str(pos_auc))
    results_file.write(", ")
    results_file.write(str(neg_auc))
    results_file.write(", ")    
    results_file.write(str(tp))
    results_file.write(", ")
    results_file.write(str(tn))
    results_file.write(", ")
    results_file.write(str(fp))
    results_file.write(", ")
    results_file.write(str(fn))
    results_file.write(", ")
    results_file.write(str(accuracy_temp_train))
    results_file.write(", ")
    results_file.write(str(precision_train))
    results_file.write(", ")
    results_file.write(str(recall_train))
    results_file.write(", ")
    results_file.write(str(FScore_train))
    results_file.write(", ")
    results_file.write(str(specificity_train))
    results_file.write(", ")
    results_file.write(str(pos_auc_train))
    results_file.write(", ")    
    results_file.write(str(neg_auc_train))
    results_file.write(", ")    
    results_file.write(str(tp_train))
    results_file.write(", ")
    results_file.write(str(tn_train))
    results_file.write(", ")
    results_file.write(str(fp_train))
    results_file.write(", ")
    results_file.write(str(fn_train))       

