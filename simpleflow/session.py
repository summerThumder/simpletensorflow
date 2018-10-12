from .operations import Operation, Variable, Placeholder,compute_gradients

class Session(object):
    def __init__(self):
        ''' Session constructor.
        '''
        # Graph the session computes for.
        # global DEFAULT_GRAPH

        self.graph = DEFAULT_GRAPH

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        ''' Context management protocal method called after `with-block`.
        '''
        self.close()

    def close(self):
        #close时把输出值归None
        all_nodes=(self.graph.constants + self.graph.variables +
                     self.graph.placeholders + self.graph.operations +
                     self.graph.trainable_variables)
        for node in all_nodes:
            node.output_value = None

    #feed_dict参数是专门给placeholder用的

    def run(self,operation, feed_dict=None):
        postorder_nodes=_get_prerequisite(operation)

        for node in postorder_nodes:
            if type(node) is Placeholder:
                node.output_value=feed_dict[node]
            else:
                node.compute_output()

   
        #operation也在node里
        return operation.output_value



#返回的顺序是第一层到最后一层
def _get_prerequisite(operation):
    postorder_nodes=[]

    def postorder_traverse(operation):
        if isinstance(operation,Operation):
             for input_node in operation.input_nodes:
                 postorder_traverse(input_node)
        postorder_nodes.append(operation)
    postorder_traverse(operation)

    return postorder_nodes

