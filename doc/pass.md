```C++

1.onnx 参数做参考
  attr不做cuda参数，但做graph.create的参数
  attr作为graph.create的参数，int类型使用64位（int_t）
  注：gather如果输入indices为constant，也设置为一个indices，binary也是

  layerNorm:
    cuda: (X, scale, bias, Y, mean, instddev)
    mlir: (X, scale, bias, axis) return (Y, mean, instddev)
  
  binary:
    cuda: (X, Y) or (X)
    mlir: (X, Y) return (Z, )
  
  elementWise:
    cuda: (X)
    mlir: (X) return (Z, )
  
  gather:
    cuda: (X, indices, Y)
    mlir: (X, indices, axis) return (Y, )

2.build.create返回值使用vector
  解决：layerNorm的算子中，callop调用，可以直接返回三个结果的组合

3.测试算子性能layerNorm与cudnn，使用forwardTainning函数，workspace为地址空间

问题：
  1.layernorm算子，保持生成的cuda代码参数列表的一致问题
    方案1：保持参数列表不变，在生成ir之前就确定好，哪些参数在kernel会使用
    问题：builder operator 的代码就需要去判断scale和bias以及mean和invStdDev是否需要
          scale和bias：
            可以直接在builder时传入null，判断cuda kernel 的scale和bias参数是否起作用
          mean和invStdDev：
            需要在buidler时传入一个flag，判断cuda kernel 的mean和invStdDev参数是否起作用
            注：若mean和invStdDev参数不起作用，那么将不会生成与其有关的ir，从而returnop就不能被构建，因为returnop需要mean和invStdDev的memref
    
```