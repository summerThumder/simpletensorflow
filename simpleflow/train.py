
from .operations import Operation,compute_gradients




class GradientDescentOptimizer(object):

    def __init__(self,learning_rate):
        self.learning_rate=learning_rate

    def minize(self,loss):

        learning_rate=self.learning_rate

        class MinimizationOperation(Operation):
            def compute_output(self):
                #loss是target_op
                #反向传播函数是在operations里定义，train里调用
                grad_table = compute_gradients(loss)

                trains={}
                for var in DEFAULT_GRAPH.trainable_variables:

                    if var in grad_table:
                        grad=grad_table[var]
                        #var不是拷贝而是地址，直接修改原地址，这个compute_out没有输出
                        var.output_value -= learning_rate * grad



                return  trains

        return MinimizationOperation()
