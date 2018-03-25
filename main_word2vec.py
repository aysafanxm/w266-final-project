import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Read features
def read_feature(file):
  print("reading feature information...\n")
  res = []
  with open(file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  for line in lines:
    line = line.split()
    for i in range(len(line)):
      line[i] = float(line[i])
    res.append(line)
  return np.array(res)

# Read labels
def read_label(file):
  print("reading label information...\n")
  res = []
  with open(file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
  for line in lines:
    line = int(line.strip())
    res.append(line)
  return np.array(res)

def addLayer(inputData, inSize, outSize, activity_function = None):  
    Weights = tf.Variable(tf.random_normal([inSize, outSize]))   
    basis = tf.Variable(tf.random_uniform([1,outSize], -1, 1))    
    weights_plus_b = tf.matmul(inputData, Weights) + basis  
    #Wx_plus_b = tf.nn.dropout(weights_plus_b, keep_prob = 0.8)     # To prevent overfitting

    if activity_function is None:  
        ans = weights_plus_b  
    else:  
        ans = activity_function(weights_plus_b)
    return ans  

def net(x_data, y_data, x_test, y_test):
    is_train = True


    insize = x_data.shape[1]
    outsize = 8
    xs = tf.placeholder(tf.float32,[None, insize]) 
    ys = tf.placeholder(tf.float32,[None, outsize]) 
    keep_prob = tf.placeholder(tf.float32)  
      
    l1 = addLayer(xs, insize, 40,activity_function=None)  
    l2 = addLayer(l1, 40, 20,activity_function=tf.nn.sigmoid)  
    l3 = addLayer(l2, 20, 10,activity_function=tf.nn.softmax)  
    l4 = addLayer(l3, 10, outsize,activity_function=tf.nn.softmax)


    y = l4
    #loss = tf.reduce_sum(tf.reduce_sum(tf.square((ys-l4)),reduction_indices = [1]))  
    #loss = -tf.reduce_mean(ys * tf.log(l3))
    #loss = tf.reduce_sum(tf.square((ys-y)))
    loss = -tf.reduce_sum(ys * tf.log(y))
    #loss = tf.reduce_sum(-tf.reduce_sum(ys * tf.log(y),reduction_indices=[1]))  # loss  
    train =  tf.train.GradientDescentOptimizer(0.00001).minimize(loss) 

    # Turn 1 dimensional label vectors to 14 dimensional vectors which has only one element = 1
    new_ydata = []
    for i in range(y_data.shape[0]):
      new_ydata.append([0]*outsize)
      new_ydata[i][y_data[i]] = 1
      # print(new_ydata[i])
    new_ydata = np.array(new_ydata)
        
    saver=tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        if is_train: 
            run_step = 4000
            for i in range(run_step):  
                sess.run(train,feed_dict={xs:x_data,ys:new_ydata})  
                if i%50 == 0:  
                    print(sess.run(loss,feed_dict={xs:x_data,ys:new_ydata}))
            # save the model
            saver=tf.train.Saver(max_to_keep=1)
            saver.save(sess,'model/net.ckpt')
        else:     # take a trained model
            saver.restore(sess, 'model/net.ckpt')
            print("save success!")

        # Prediction
        res = sess.run(fetches=y, feed_dict={xs: x_test})
        new_res = []
        for ele in res:
            mmax = -1111
            index = -1
            for i in range(outsize):
                if ele[i] > mmax:
                    index, mmax  = i, ele[i]
            new_res.append(index)
        #print(new_res)
        new_res = np.array(new_res)
        counter = 0
        for i in range(len(new_res)):
          if(y_test[i] == new_res[i]):
            counter += 1
        print("Accuracy: ", counter/len(new_res))
        print(classification_report(new_res, y_test))

def main():
  feature = read_feature('data/feature.txt')
  label = read_label('data/label.txt')

  x_train , x_test , y_train , y_test = train_test_split(feature, label, test_size = 0.1,random_state=0)
  net(x_train, y_train, x_test, y_test)

if __name__ == '__main__':
  main()