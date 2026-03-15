def load_array(data_arrays, batch_size, is_train=True):  #@save     
  """构造一个PyTorch数据迭代器"""     
  dataset = data.TensorDataset(*data_arrays)     
  return data.DataLoader(dataset, batch_size, shuffle=is_train)
  
  batch_size = 10 
  data_iter = load_array((features, labels), batch_size)
  
  next(iter(data_iter))
  [tensor([[ 0.1554, -0.2034],
         [-0.2140,  1.0352],
         [-0.4209,  0.0428],
         [ 0.1887,  0.6141],
         [ 0.4987, -0.2314],
         [ 0.0653,  1.6406],
         [-1.1881,  0.2900],
         [-0.2824,  0.5910],
         [ 0.9963, -0.1816],
         [-1.6830, -1.3963]]),
 tensor([[ 5.2116],
         [ 0.2479],
         [ 3.2188],
         [ 2.4845],
         [ 5.9884],
         [-1.2453],
         [ 0.8441],
         [ 1.6217],
         [ 6.8072],
         [ 5.5692]])]
