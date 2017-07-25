import keras.layers
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
from scipy.misc import imread
from sklearn.utils import shuffle
from skimage.transform import rescale
import os


def listdir(path):
    l = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            l.append(f)
    return l

print('Sucessfully imported all modules')


def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

input_shape = (230, 230, 3)
left_input = Input(input_shape)
right_input = Input(input_shape)
#build convnet to use in each siamese 'leg'
convnet = Sequential()
convnet.add(Conv2D(64,(10,10),activation='relu',input_shape=input_shape,
                   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(7,7),activation='relu',
                   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(256,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(Flatten())
convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))
#encode each of the two inputs into a vector with the convnet
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
#merge two encoded inputs with the l1 distance between them
L1_distance = lambda x: K.abs(x[0]-x[1])
both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(both)
siamese_net = Model(input=[left_input,right_input],output=prediction)
#optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)

optimizer = Adam(0.00006)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

siamese_net.count_params()


class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    #path to data folder
    #shape - shape of the images
    def __init__(self,path,shape):
        self.data = {}
        self.categories = {}
        self.w,self.h = shape
        #self.n_val,self.n_ex_val,_,_ = self.data['val'].shape
        self.path = path

        
    def load_rnd_image(self, category, data_path):
        category_path = os.path.join(data_path, category)
        try:
            idx = rng.randint(len(listdir(category_path)))
        except ValueError:
            print(category)
            print(data_path)
        example = listdir(category_path)[idx]
        example_path = os.path.join(category_path, example)
        return imread(example_path,mode = 'RGB')/255 #rescale(imread(example_path,mode = 'RGB'), 1)#/255 #TODO: Remove reshape
        
        
    def get_batch(self,n,s="train"):
        """Create batch of n pairs, half same class, half different class"""
        data_path = os.path.join(self.path,s)
        n_classes = len(listdir(data_path))
        #select random categories by name
        categories_names = np.array(listdir(data_path))
        try:
            categories_n = rng.choice(n_classes,size=(n,),replace=False)
        except ValueError:
            print(data_path)
            print(n_classes)
        categories = categories_names[categories_n]
        pairs=[np.zeros((n, self.h, self.w, 3), dtype = np.float32) for i in range(2)]
        targets=np.zeros((n,))
        targets[n//2:] = 1
        for i in range(n):
            category = categories[i]
            category_n = categories_n[i]
            pairs[0][i,:,:,:] = self.load_rnd_image(category, data_path)
            #pick images of same class for 1st half, different for 2nd
            category_2 = category if i >= n//2 else categories_names[(category_n + rng.randint(1,n_classes-1)) % n_classes]
            pairs[1][i,:,:,:] = self.load_rnd_image(category_2, data_path)
        return pairs, targets
    
    
    def make_oneshot_task(self,N,s='val', verbose = False):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        data_path = os.path.join(self.path,s)
        n_classes = len(listdir(data_path))
        #get list of category names
        categories_names = np.array(listdir(data_path))
        #select random numbers for rnadom category selection 
        categories_n = rng.choice(n_classes,size=(N,),replace=False)
        #select random category names
        categories = categories_names[categories_n]
        true_category = categories[0]
        if verbose:
            print('True cateogry', ' : ' ,true_category)
        test_image = np.asarray([self.load_rnd_image(true_category, data_path)]*N).reshape(N,self.w,self.h,3)
        support_set = np.zeros((N, self.w, self.h, 3))
        for i in range(1,N):
            if verbose:
                print(i,' : ',categories[i])
            support_set[i,:,:,:] = self.load_rnd_image(categories[i], data_path)
        support_set[0,:,:] = self.load_rnd_image(true_category, data_path)
        support_set = support_set.reshape(N,self.w,self.h,3)
        targets = np.zeros((N,))
        targets[0] = 1
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image,support_set]
        return pairs, targets

    
    def test_oneshot(self,model,N,k,verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        pass
        n_correct = 0
        if verbose:
            print("Evaluating model on {} unique {} way one-shot learning tasks ...".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N)
            probs = model.predict(inputs)
            if np.argmax(probs) == 0:
                n_correct+=1
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct
    
    

loader = Siamese_Loader(path = '', shape = (230,230))
print('Created the Loader object')


#Training loop
evaluate_every = 2000
loss_every=50
batch_size = 20
N_way = 20
n_val = 250
#siamese_net.load_weights("/home/soren/keras-oneshot/weights")
with open('result.txt', "w") as f:
    pass
with open('loss.txt', "w") as f:
    pass

with open('result.txt', 'a') as f:
    f.write('-----------------')
with open('loss.txt', 'a') as f:
    f.write('-----------------')

best = 0.0001
max_epochs = evaluate_every*20
print('Started Training')
for i in range(1,max_epochs):
    print('Batch ', i)
    (inputs,targets)=loader.get_batch(batch_size)
    loss=siamese_net.train_on_batch(inputs,targets)
    if i % evaluate_every == 0:
        print('Testing accuracy')
        val_acc = loader.test_oneshot(siamese_net,N_way,n_val,verbose=True)
        if val_acc >= best:
            print("saving")
            siamese_net.save('/weights')
            with open('result.txt', 'a') as f:
                f.write(str(val_acc)+',' + str(i) + '\n')
            best=val_acc

    if i % loss_every == 0:
        print("iteration {",i,"}, and loss is: ", loss)
        with open('loss.txt', 'a') as f:
            f.write(str(loss)+',' + str(i) + '\n')


        

