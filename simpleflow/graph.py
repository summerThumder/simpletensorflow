
#用于with代码块
class Graph(object):
    #构造的时候全空图
    def __init__(self):
        self.operations, self.constants, self.placeholders = [], [], []
        self.variables, self.trainable_variables = [], []

    def __enter__(self):
        ''' Reset default graph.
        '''
        #enter是进入with代码块的意思

        global  DEFAULT_GRAPH
        self.old_graph = DEFAULT_GRAPH
        DEFAULT_GRAPH = self
        return self


    #这些变量有什么用
    def __exit__(self, exc_type, exc_value, exc_tb):
        ''' Recover default graph.
        '''
        #退出with时恢复旧的图
        global DEFAULT_GRAPH
        DEFAULT_GRAPH = self.old_graph

    #返回对象自身，没有也可以
    def as_default(self):
        ''' Set this graph as global default graph.
        '''
        return self
