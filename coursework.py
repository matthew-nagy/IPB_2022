import numpy as np
import matplotlib.pyplot as plt
#the labels I am keeping are 
# # T-shirt/top(0)
# # Trouser(1)
# # Dress(3)
# # Bag(8)
# # Ankle boo(9)
my_kept_labels = [0, 1, 3, 8, 9]

from math import log2

import time
duration = time.perf_counter()

def get_duration():
    global duration
    n= time.perf_counter()
    dif = n - duration
    duration = n
    return dif

benchmarking = False
def benchmark(place):
    global benchmarking
    if benchmarking:
        print(">" + place + ": ", get_duration())

import utils.mnist_reader as mnist_reader

def get_simplified_data(data: list, labels: list, accepted_labels: list):
    simple_data = []

    for i in range(len(data)):
        if labels[i] in accepted_labels:
            datum = [float(j) / 255.0 for j in data[i]]
            #Make a one hot vector with the chosen class set to 1
            label = np.zeros(len(my_kept_labels))
            label[my_kept_labels.index(labels[i])] = 1

            simple_data.append((datum, label, my_kept_labels.index(labels[i])))

    return simple_data

#Given some dataset array with inputs ([0]) and targets ([1]), will return 2 @param batch_size 
#lists, one of inputs and one of targets
def generate_batch(dataset: list, special_dataset, batch_size: int):
    inputs = special_dataset[0]#np.vstack([ex[0] for ex in dataset])
    targets = special_dataset[1]#np.vstack([ex[1] for ex in dataset])
    
    rand_inds = np.random.randint(0, len(dataset), batch_size)
    inputs_batch = inputs[rand_inds]
    targets_batch = targets[rand_inds]
    
    return inputs_batch, targets_batch


def sigmoid(a):
    denominator = 1.0 + (2.71828 ** (a * - 1.0))
    return 1.0 / denominator
    

class nn_one_layer():
    def __init__(self, input_size, hidden_size, output_size):
        #define the input/output weights W1, W2
        self.W1 = 0.1 * np.random.randn(input_size, hidden_size)
        self.W2 = 0.1 * np.random.randn(hidden_size, output_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.last_input = None
        self.last_hidden = None
        self.last_output = None
        
        self.f = sigmoid
    
    #for matrix multiplication use np.matmul()
    def forward(self, u: list):
        z = np.matmul(u, self.W1)
        h = self.f(z)
        v = self.f(np.matmul(h, self.W2))

        return v, h, z

    def write_network(self, name):
        with open(name, "w") as file:
            # Write bytes to file
            file.write(str(self.input_size))
            file.write('\n')
            file.write(str(self.hidden_size))
            file.write('\n')
            file.write(str(self.output_size))
            file.write('\n')
            for i in self.W1:
                for j in i:
                    file.write(str(j))
                    file.write('\n')
            for i in self.W2:
                for j in i:
                    file.write(str(j))
                    file.write('\n')
    
    def read_network(self, name):
        with open(name, "r") as file:
            self.input_size = int(file.readline())
            self.hidden_size = int(file.readline())
            self.output_size = int(file.readline())
            
            self.W1 = 0.1 * np.random.randn(self.input_size, self.hidden_size)
            self.W2 = 0.1 * np.random.randn(self.hidden_size, self.output_size)
            
            for i in range(len(self.W1)):
                for j in range(len(self.W1[i])):
                    self.W1[i][j] = float(file.readline())
            for i in range(len(self.W2)):
                for j in range(len(self.W2[i])):
                    self.W2[i][j] = float(file.readline())



#loss function as defined above
def loss_mse(preds, targets):
    loss = np.sum((preds - targets)**2)
    return 0.5 * loss

#derivative of loss function with respect to predictions
def loss_deriv(preds, targets):
    dL_dPred = preds - targets
    return dL_dPred

#derivative of the sigmoid function
def sigmoid_prime(a):
    dsigmoid_da = sigmoid(a)*(1-sigmoid(a))
    return dsigmoid_da

#compute the derivative of the loss wrt network weights W1 and W2
#dL_dPred is (precomputed) derivative of loss wrt network prediction
#X is (batch) input to network, H is (batch) activity at hidden layer
def backprop(W1, W2, dL_dPred, U, H, Z):
    #hints: for dL_dW1 compute dL_dH, dL_dZ first.
    #for transpose of numpy array A use A.T
    #for element-wise multiplication use A*B or np.multiply(A,B)
    
    dL_dW2 = np.matmul(H.T, dL_dPred)
    dL_dH = np.matmul(dL_dPred, W2.T)
    dL_dZ = np.multiply(sigmoid_prime(Z), dL_dH)
    dL_dW1 = np.matmul(U.T, dL_dZ)
    
    return dL_dW1, dL_dW2

#train the provided network with one batch according to the dataset
#return the loss for the batch
def train_one_batch(nn, dataset, special_dataset, batch_size, lr):
    inputs, targets = generate_batch(dataset,special_dataset, batch_size)
    preds, H, Z = nn.forward(inputs)

    loss = loss_mse(preds, targets)

    dL_dPred = loss_deriv(preds, targets)
    dL_dW1, dL_dW2 = backprop(nn.W1, nn.W2, dL_dPred, U=inputs, H=H, Z=Z)

    nn.W1 -= lr * dL_dW1
    nn.W2 -= lr * dL_dW2
    
    return loss

#test the network on a given dataset
def test(nn, dataset, batch_size):
    inputs, targets = generate_batch(dataset, batch_size=batch_size)
    preds, H, Z = nn.forward(inputs) 
    loss = loss_mse(preds, targets)
    return loss


def shannon(prob_x):
    return -1.0 * [i * log2(i) for i in prob_x]

def bound_activations(x):
    avg = sum(x) / float(len(x))
    x[x >= avg] = 1.0
    x[x < avg] = 0.0
    return x

#A 2d list
#Index 1 is class label
#Index 2 is neuron index
def cond_shannon(prob_of_neuron_given_class: list[list[float]]):
    tot = 0.0
    
    for values in prob_of_neuron_given_class:
        for val in values:
            tot += val * log2(val)
    return tot * -1.0

import multiprocessing

def shannon_worker(nn: nn_one_layer, test_data: list[float], incr: float) -> tuple[int, float, float]:
    hs = dict()
    os = dict()
    for data in test_data:
        output, hidden, _ = nn.forward(data)

        hidden = bound_activations(hidden)
        max_out = max(output)
        for i in range(len(output)):
            if output[i] == max_out:
                output[i] = 1.0
            else:
                output[i] = 0.0
        

        hid_str = str(hidden)
        out_str = str(output)

        if hid_str in hs:
            hs[hid_str] += incr
        else:
            hs[hid_str] = incr
        
        if out_str in os:
            os[out_str] += incr
        else:
            os[out_str] = incr
    
    h_tot = sum([i * log2(i) for i in hs.values()])
    o_tot = sum([i * log2(i) for i in os.values()])

    return h_tot, o_tot

def get_shannons(nn: nn_one_layer, sorted_test_data: list[list[float]], test_dat_len: int, num_of_classes: int) -> tuple[float, float]:

    incr = 1.0 / float(test_dat_len)
    pool = multiprocessing.Pool(5)
    results = []

    for data in sorted_test_data:
        results.append(pool.apply_async(func=shannon_worker, args=(nn, data, incr, )))
    pool.close()
    pool.join()

    h_tot = 0
    o_tot = 0
    for res in results:
        r = res.get(None)
        h_tot += r[0]
        o_tot += r[1]
    return h_tot * -1.0, o_tot * -1.0
    
def main():
    full_train_data, full_train_labels = mnist_reader.load_mnist('data/fashion', kind='train')
    full_test_data, full_test_labels = mnist_reader.load_mnist('data/fashion', kind='t10k')
    train_data = get_simplified_data(full_train_data, full_train_labels, my_kept_labels)
    test_data = get_simplified_data(full_test_data, full_test_labels, my_kept_labels)


    input_size = 784#28 * 28
    hidden_size = 300
    output_size = 5

    nn = nn_one_layer(input_size, hidden_size, output_size)

    chosen_dataset = train_data

    batch_size = 30 #number of examples per batch
    nbatches = 200 #number of batches used for training
    lr = 0.1 #learning rate

    losses = [] #training losses to record
    test_losses = []
    hidden_shannon=[]
    output_shannon=[]

    special_dataset = (np.vstack([ex[0] for ex in chosen_dataset]), np.vstack([ex[1] for ex in chosen_dataset]))

    sorted_test_data = [[] for i in range(len(my_kept_labels))]
    for (dat, one_hot, label) in test_data:
        sorted_test_data[label].append(dat)

    benchmark("Time to start")
    for i in range(nbatches):
        get_duration()
        loss = train_one_batch(nn, chosen_dataset, special_dataset, batch_size=batch_size, lr=lr)
        losses.append(loss)
        benchmark("Ran training")

        #test_loss = test(nn, test_data, batch_size)
        #test_losses.append(test_loss)

        if i % 1 == 0:
            shan_h, shan_o = get_shannons(nn, sorted_test_data, len(test_data), len(my_kept_labels))
            benchmark("Running shannons")
            hidden_shannon.append(shan_h)
            output_shannon.append(shan_o)

            print(i,"  ", loss, "  ", shan_h, "  ", shan_o)
        #shannons.append(0)
        #print(i, "  ", loss)

    fig, axs = plt.subplots(3)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(np.arange(1, nbatches+1), losses)
    axs[1].plot(np.arange(1, len(hidden_shannon)+1), hidden_shannon)
    axs[2].plot(np.arange(1, len(output_shannon)+1), output_shannon)
    plt.show()

    name = 'nem.txt'
    nn.write_network(name)

def a_salad_with(ingredient):
    return "I would like a salad with " + ingredient + " please!"

def p_result(res):
    print(res)

if __name__ == "__main__":
    #pool = multiprocessing.Pool(5)
    #pool.apply_async(func=a_salad_with, args=("ham",), callback=p_result)
    #pool.close()
    #pool.join()

    main()
    #main2()
'''
output type (sigmoid, normal)	os/on
number of hidden neurons
run number with these settings
batch size
learning rate (0.1 = zp1, 0.05 = zpz5)
training or loss curve, t, l, tl
'''

'''
epoch_list = []
training_loss_list = []
test_loss_list = []
for i in range(epochs):
    train_loss = train_one_batch(nn, train_data, batch_size, learning_rate)
    training_loss_list.append(train_loss)

    test_loss = test(nn, test_data, batch_size)
    test_loss_list.append(test_loss)

    epoch_list.append(i)
    print(i,"\t", train_loss, "\t", test_loss)
'''

'''
plt.scatter(epoch_list, training_loss_list)
plt.scatter(epoch_list, test_loss_list, c="orange")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
'''