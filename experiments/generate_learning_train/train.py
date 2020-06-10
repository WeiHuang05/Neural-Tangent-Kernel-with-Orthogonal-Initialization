import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from time import time
import math
from include.data import get_data_set
from include.model import model, lr
from sys import argv

script, idex, learning_number = argv
idx = int(idex)
lri = int(learning_number)

# PARAMS
_BATCH_SIZE = 1000
_EPOCH = 20
_SAVE_PATH = "./tensorboard/cifar-10-v1/"

_MARK = 0

def train(lri, epoch):
    global epoch_start
    global _MARK
    epoch_start = time()
    batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
    i_global = 0
    print ("real learning rate is: {}",format(lr(lri)))

    new_order=np.random.permutation(len(train_x))
    train_x1 = train_x[new_order]
    train_y1 = train_y[new_order]
    for s in range(batch_size):
        if _MARK ==1:
            print ('exit batch loop now')
            break
        batch_xs = train_x1[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
        batch_ys = train_y1[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]

        start_time = time()
        i_global, _, batch_loss, batch_acc = sess.run(
            [global_step, optimizer, loss, accuracy],
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(lri)})
        duration = time() - start_time
        
        msg = "Global step: {:>5} - acc: {:.4f} - loss: {:.4f} - {:.1f} sample/sec"
        print(msg.format(i_global, batch_acc, batch_loss, _BATCH_SIZE / duration))
        #test_and_save(i_global, epoch, lri)

        if batch_acc > 0.25:
            
            _MARK = 1
            file_.write(str(lri)+' ' + str(lr(lri)) + ' ' + str(i_global)+ ' ' + str(batch_acc)+'\n')
        
               

def test_and_save(_global_step, epoch, lri):
    global global_accuracy
    global epoch_start

    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(
            y_pred_cls,
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(lri)}
        )
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    if acc>25:
        global _MARK
        _MARK = 1
        file_.write(str(lri) + ' ' +str(lr(lri)) +' ' + str(_global_step) +' '+ str(acc) + '\n')
       
    correct_numbers = correct.sum()

    hours, rem = divmod(time() - epoch_start, 3600)
    minutes, seconds = divmod(rem, 60)
    mes = " accuracy: {:.2f}% ({}/{}) - time: {:0>2}:{:0>2}:{:05.2f}"
    print(mes.format( acc, correct_numbers, len(test_x), int(hours), int(minutes), seconds))



train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")
tf.set_random_seed(idx)
train_start = time()

 

filename = 'tanh2_gaussian_L50_N400_lr_%d' %(idx) 
file_= open("{}.csv".format(filename),'a+')  

#global _MARK 
#_MARK = 0
sess = tf.Session()
global_accuracy = 0
epoch_start = 0 
x, y, output, y_pred_cls, global_step, learning_rate = model()
# loss = tf.reduce_mean(tf.reduce_sum((y-output)**2,reduction_indices=[1]))
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(output),reduction_indices=[1])) 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)                                
correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess.run(init)

print ("\n Learning rate:{}". format(lri))
        
for i in range(_EPOCH):
	if _MARK == 1:
		print ('exist epoch loop now')
		break
	print("\nEpoch: {}/{}\n".format((i+1), _EPOCH))
	train(lri, i)

hours, rem = divmod(time() - train_start, 3600)
minutes, seconds = divmod(rem, 60)
mes = "Best accuracy pre session: {:.2f}, time: {:0>2}:{:0>2}:{:05.2f}"
print(mes.format(global_accuracy, int(hours), int(minutes), seconds))

   
