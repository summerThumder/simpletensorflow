import numpy as np
from queue import Queue


class Operation(object):
    ''' Base class for all operations in simpleflow.
     An operation is a node in computational graph receiving zero or more nodes
     as input and produce zero or more nodes as output. Vertices could be an
     operation, variable or placeholder.
     '''
    def __init__(self,*input_nodes,name=None):
        self.input_nodes=input_nodes
        self.output_nodes=[]
        self.output_value=None
        self.name=name
        self.graph=DEFAULT_GRAPH

        #给每个输入节点的输出加上这个op
        for node in input_nodes:
            node.output_nodes.append(self)

        self.graph.operations.append(self)

    def compute_output(self):
        raise NotImplementedError

    def compute_gradient(self,grad=None):
        raise NotImplementedError

class Add(Operation):
    def __init__(self,x,y,name=None):
        super(self.__class__,self).__init__(x,y,name=name)

    def compute_output(self):
        x,y=self.input_nodes
        self.output_value=np.add(x.output_value,y.output_value)
        return self.output_value

    def compute_gradient(self,grad=None):
        ''' Compute the gradients for this operation wrt input values.
                :param grad: The gradient of other operation wrt the addition output.
                :type grad: number or a ndarray, default value is 1.0.
         '''
        x,y=[node.output_value for node in self.input_nodes]

        if grad is None:
            grad=np.ones_like(self.output_value)

        grad_wrt_x=grad
        #############把grad浓缩至与x相同维数
        while np.ndim(grad_wrt_x)>len(np.shape(x)):
            grad_wrt_x=np.sum(grad_wrt_x,axis=0)

        for axis,size in enumerate(np.shape(x)):
              if size==1:
                  grad_wrt_x=np.sum(grad_wrt_x,axis=axis)

        grad_wrt_y=grad
        while np.ndim(grad_wrt_y) > len(np.shape(y)):
            grad_wrt_y = np.sum(grad_wrt_y, axis=0)
        for axis, size in enumerate(np.shape(y)):
            if size == 1:
                grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)

        return [grad_wrt_x,grad_wrt_y]


def add(x, y, name=None):
    ''' Returns x + y element-wise.
    '''
    return Add(x, y, name)

#---------------------------------------------------------------------
class Square(Operation):
    ''' Square operation.
    '''
    def __init__(self, x, name=None):
        ''' Operation constructor.
        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.
        :param name: The name of the operation.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        ''' Compute and return the value of square function.
        '''
        x, = self.input_nodes
        self.output_value = np.square(x.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        ''' Compute the gradient for square operation wrt input value.
        :param grad: The gradient of other operation wrt the square output.
        :type grad: ndarray.
        '''
        input_value = self.input_nodes[0].output_value

        if grad is None:
            grad = np.ones_like(self.output_value)

        return grad*np.multiply(2.0, input_value)

def square(x, name=None):
    ''' Computes square of x element-wise.
    '''
    return Square(x, name=name)
#---------------------------------------------------------------------------

class Constant(object):
    def __init__(self,value,name):
        self.value=value
        self.output_value=None
        self.output_nodes=[]
        self.name=name
        DEFAULT_GRAPH.constants.append(self)

    def compute_output(self):
        #const的输出一开始是None,但初始化时传进来一个value
        if self.output_value is None:
            self.output_value=self.value
        return self.output_value

    def __add__(self,other):
        return Add(self,other)

def constant(value, name=None):
    ''' Create a constant node.
    '''
    return Constant(value, name=name)


#----------------------------------------------
class Variable(object):
    def __init__(self,initial_value=None,name=None,trainable=True):
        self.initial_value=initial_value
        self.output_value=None
        self.output_nodes=[]
        self.name=name
        self.graph = DEFAULT_GRAPH
        self.graph.variables.append(self)
        if trainable:
            self.graph.trainable_variables.append(self)

    def compute_output(self):
        #与const类似
       if self.output_value is None:
           self.output_value=self.initial_value
       return self.output_value

    def __add__(self, other):
        return Add(self, other)

def variable(value,name=None):
    return Variable(value, name=name)


class Placeholder(object):
    def __init__(self,name=None):
       self.output_value=None
       self.output_nodes=[]
       self.name=name
       self.graph = DEFAULT_GRAPH
       self.graph.placeholders.append(self)

    def __add__(self, other):
        return Add(self, other)

def placeholder(name=None):
    return Placeholder(name=name)


#这个函数传的参数的含义？应该是最后一层的op
def compute_gradients(target_op):
    ''' Backpropagation implementation computing gradient of target operation wrt
            all the other connected nodes.
        :param target_op: The target operation whose gradient wrt other nodes would
                          be computed.
        :type target_op: Any operation type.
        :return grad_table: A table containing node objects and gradients.
        :type grad_table: dict.
        '''
    grad_table={}
    grad_table[target_op]=np.ones_like(target_op.output_value)

    # Perform a breadth-first search staring from the target_op in graph.
    # Queue for node traverasl.
    queue=Queue()
    queue.put(target_op)

    # Set for visited nodes.
    visited = set()
    visited.add(target_op)

    while not queue.empty():
        node=queue.get()

        if node !=target_op:
            grads_wrt_node_output=[]


            for output_node in node.output_nodes:
                #gradtable op:梯度
                grad_wrt_output_node_output = grad_table[output_node]

                #compute_gradient需要传入梯度,返回一个wrt数组
                #反向传播用当前节点的output_node计算梯度，grad_wrt_node_output是反向输出的wrt梯度
                grad_wrt_node_output = output_node.compute_gradient(grad_wrt_output_node_output)
                if len(output_node.input_nodes)>1:
                    input_node_index=output_node.input_nodes.index(node)
                    #前一个是grads,后一个是grad
                    grads_wrt_node_output.append(grad_wrt_node_output[input_node_index])
                else:
                    grads_wrt_node_output.append(grad_wrt_node_output)

            tot_grad_wrt_node_output=sum(grads_wrt_node_output)
            grad_table[node]=tot_grad_wrt_node_output
                    # Put adjecent nodes to queue.
                    #因为反向传播，所以是input_nodes
        if hasattr(node,'input_nodes'):
            for input_node in node.input_nodes:
                #考虑1->2->3,1->3的shortcut如不加这句1会计算两次
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table

