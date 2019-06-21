import numpy as np


class Network(object):

    def __init__(self,layers):
	"""
        layers网络的层数包括输入，输出，隐藏层
        bias 网络权重的偏执，输入层是没有bias，bias数量等于神经元的数量
	weights，这是一个全连神经网络，weights 维度等于前一层的神经元数量 * 后面一层神经元的数量
        """        
        self.layers = layers
        self.size = len(layers)
        self.bias = [np.random.randn(x,1) for x in layers[1:]]
        self.weights = [np.random.randn(x,y) for x,y in zip(layers[:-1],layers[1:])]


    def feedforward(self,a):
        """
        前向传播，最终输出y
        """
        for index in range(len(self.bias)):
            """
            换一种遍历方式zip(self.bias,self.weights),更加直白，易读
            """
            a = sigmoid(np.dot(a,w[index]+b[index]))
        return a
    def SGD(self,train_data,mini_batchsize=32,epoch=10,alpha=0.001,test_data=None):
        """

        """
        if test_data:
        	test_len = len(test_data)

        n = len(train_data)
        for i in range(epoch):
            np.random.shuffle(train_data)
            mini_batchs = [train_data[start:start+mini_batchsize] for start in range(0,n,mini_batch_size)]
            for mini_batch in mini_batchs:
                self.updata_minibatch(mini_batch,alpha)

            if test_data:
            	print("epoch{0} {1}/{2} complete".fromat(i,self.􏾥􏿁􏾖􏾘􏾷􏾖􏾑􏾥􏾥􏿁􏾖􏾘􏾷􏾖􏾑􏾥evaluate(test_data),test_len))
            else:
            	print("epoch{0} complete".fromat(i))
    def update_minibatch(self,mini_batch,alpha):
        
    	total_w = [np.zeros(w.shape) for w in self.weights]
    	total_b = [np.zeros(b.shape) for b in self.bias]

    	for x,y in mini_batch:
    		"""
			相当于求和
    		"""
    		delta_w ,delta_b = self.backprop(x,y)

    		total_b = [t_b + d_b for t_b,d_b in zip(total_b,delta_b)]

    		total_w = [t_w + d_w for t_w,d_w in zip(total_w,delta_w)]

    	self.weights = [w - (alpha/len(mini_batch) * t_w) for w,t_w in zip(self.weights,total_w)]
		self.weights = [b - (alpha/len(mini_batch) * t_b) for b,t_b in zip(self.bias,total_b)]
	
	def backprop(self,x,y):

		n_b = [np.zeros(b.shape) for b in self.bias]
		n_w = [np.zeros(w.shape) for w in self.weights]

		activation = x
		activations = [x]
		zs = []

		for b,w in zip(self.bias,self.weights):

			z = np.dot(w,activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# back
		delta = self.cost_derivative(activations[-1],y) * sigmoid_prime(zs[-1]) 
		#  (y - y.) * h'(w)
		#  deltw = del_b * x
		n_b[-1] = delta
		n_w[-1] = np.dot(delta,activations[-2].transpose())

		for l in range(2,self.size):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(),delta) * sp
			t_b[-l] = delta
			t_w[-l] = np.dot(delta,activations[-l-1].transpose())
		return (n_b,n_w)
    def 􏾥􏿁􏾖􏾘􏾷􏾖􏾑􏾥􏾥􏿁􏾖􏾘evaluate(self,data):
    	test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in data]
    	return sum(int(x == y) for (x,y) in test_results)
    def cost_derivative(self,output_actions,y):
    	return output_actions-y









def sigmoid(z):

    return 1.0/(1 + np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)* (1- sigmoid(z))



