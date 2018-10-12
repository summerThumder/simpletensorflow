import simpleflow as sf
# 测试constant前向传播
with sf.Graph().as_default():
    a = sf.constant(1.0, name='a')
    b = sf.constant(2.0, name='b')
    result = sf.add(a, b, name='result')
    # Create a session to compute
    with sf.Session() as sess:
        print(sess.run(result))

#测试placeholder前向传播
with sf.Graph().as_default():
    a = sf.placeholder()
    b = sf.placeholder()
    result = sf.add(a, b,name='result')
    # Create a session to compute
    with sf.Session() as sess:
        print(sess.run(result,feed_dict={a:10,b:30}))

#测试反向传播
max_step=100
with sf.Graph().as_default():
    a = sf.constant(1.0,name='a')
    b = sf.variable(0.0,name='b')
    c = sf.add(a, b,name='c')
    re=sf.square(c,name='re')
    opt = sf.GradientDescentOptimizer(learning_rate=0.1)
    train_op = opt.minize(re)
    # Create a session to compute
    with sf.Session() as sess:
        for step in range(max_step-1):
            print('re:',sess.run(re))
            sess.run(train_op)
            print('b:',b.output_value)

