```C++

func.func @Matmul_m4096n2048k1024(%arg0: memref<4096x1024xf32, 1>, %arg1: memref<1024x2048xf32, 1>, %arg2: memref<4096x2048xf32, 1>) -> memref<4096x2048xf32, 1> attributes {func.state = "gpu"} {
    affine.parallel (%arg3, %arg4) = (0, 0) to (32, 16) {
      %4 = memref.alloc() : memref<2x8x128xf32, 2>
      %5 = memref.alloc() : memref<2x8x128xf32, 2>
      %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg3)
      %7 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg4)
      affine.parallel (%arg5, %arg6) = (0, 0) to (16, 16) {
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c0_0 = arith.constant 0 : index
        %c7 = arith.constant 7 : index
        %c0_1 = arith.constant 0 : index
        %c0_2 = arith.constant 0 : index
        %c0_3 = arith.constant 0 : index
        %c0_4 = arith.constant 0 : index
        %8 = memref.alloc() : memref<4xf32, 3>
        %9 = memref.alloc() : memref<4xf32, 3>
        %10 = memref.alloc() : memref<2x8xf32, 3>
        %11 = memref.alloc() : memref<2x8xf32, 3>
        %12 = memref.alloc() : memref<8x8xf32, 3>
        %13 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg5)
        %14 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg6)
        // 将存结果的8*8reg初始值设置为0
        affine.for %arg7 = 0 to 8 {
          affine.for %arg8 = 0 to 8 {
            %cst = arith.constant 0.000000e+00 : f32
            affine.store %cst, %12[%arg7, %arg8] : memref<8x8xf32, 3>
          } {affine.loop = "unroll"}
        } {affine.loop = "unroll"}
        // 将数据从global转到reg 4*4
        %15 = affine.vector_load %arg0[%arg5 * 8 + %arg6 floordiv 2 + %c0 * 128 + %arg3 * 128, (%arg6 mod 2) * 4 + %c0_3] : memref<4096x1024xf32, 1>, vector<4xf32>
        affine.vector_store %15, %8[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
        %16 = affine.vector_load %arg1[(%arg5 * 16 + %arg6) floordiv 32 + %c0 * 8 + %c0_4, ((%arg5 * 16 + %arg6) mod 32) * 4 + %arg4 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
        affine.vector_store %16, %9[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
        // 将reg 转到 shared
        affine.for %arg7 = 0 to 4 {
          %22 = affine.vector_load %8[%c0 * 4 + %arg7] : memref<4xf32, 3>, vector<1xf32>
          affine.vector_store %22, %4[0, (%arg6 mod 2) * 4 + %arg7, %arg5 * 8 + %arg6 floordiv 2 + %c0 * 128] : memref<2x8x128xf32, 2>, vector<1xf32>
        } {affine.loop = "unroll"}
        %17 = affine.vector_load %9[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
        affine.vector_store %17, %5[0, (%arg5 * 16 + %arg6) floordiv 32 + %c0 * 8, ((%arg5 * 16 + %arg6) mod 32) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        gpu.barrier  // 同步

        // 将sharedA 转到 reg 2*8的 0dim
        %18 = affine.vector_load %4[(%c0_0 floordiv 8) mod 2, %c0_1, (((%arg5 * 16 + %arg6) mod 32) floordiv 4 + (((%arg5 * 16 + %arg6) floordiv 32) floordiv 4 + %c0 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %18, %10[0, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        %19 = affine.vector_load %4[(%c0_0 floordiv 8) mod 2, %c0_1, (((%arg5 * 16 + %arg6) mod 32) floordiv 4 + (((%arg5 * 16 + %arg6) floordiv 32) floordiv 4 + %c1 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %19, %10[0, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        // 将sharedB 转到 reg 2*8的 0dim
        %20 = affine.vector_load %5[(%c0 floordiv 8) mod 2, %c0_2, (%arg6 mod 4 + (((%arg5 * 16 + %arg6) floordiv 32) mod 4 + %c0 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %20, %11[0, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        %21 = affine.vector_load %5[(%c0 floordiv 8) mod 2, %c0_2, (%arg6 mod 4 + (((%arg5 * 16 + %arg6) floordiv 32) mod 4 + %c1 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %21, %11[0, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>

        // 
        affine.for %arg7 = 0 to 1024 step 8 {
          affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg7) {
            %26 = affine.vector_load %arg0[%arg5 * 8 + %arg6 floordiv 2 + %c0 * 128 + %arg3 * 128, (%arg6 mod 2) * 4 + %arg7 + 8] : memref<4096x1024xf32, 1>, vector<4xf32>
            affine.vector_store %26, %8[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
            affine.vector_store %27, %9[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
          }
          affine.for %arg8 = 0 to 7 {
            %26 = affine.vector_load %4[(%arg7 floordiv 8) mod 2, %arg8 + 1, (((%arg5 * 16 + %arg6) mod 32) floordiv 4 + (((%arg5 * 16 + %arg6) floordiv 32) floordiv 4 + %c0 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
            affine.vector_store %26, %10[(%arg8 + 1) mod 2, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
            %27 = affine.vector_load %4[(%arg7 floordiv 8) mod 2, %arg8 + 1, (((%arg5 * 16 + %arg6) mod 32) floordiv 4 + (((%arg5 * 16 + %arg6) floordiv 32) floordiv 4 + %c1 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
            affine.vector_store %27, %10[(%arg8 + 1) mod 2, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
            %28 = affine.vector_load %5[(%arg7 floordiv 8) mod 2, %arg8 + 1, (%arg6 mod 4 + (((%arg5 * 16 + %arg6) floordiv 32) mod 4 + %c0 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
            affine.vector_store %28, %11[(%arg8 + 1) mod 2, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
            %29 = affine.vector_load %5[(%arg7 floordiv 8) mod 2, %arg8 + 1, (%arg6 mod 4 + (((%arg5 * 16 + %arg6) floordiv 32) mod 4 + %c1 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
            affine.vector_store %29, %11[(%arg8 + 1) mod 2, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>

            affine.for %arg9 = 0 to 8 {
              affine.for %arg10 = 0 to 8 {
                %30 = affine.load %12[%arg9, %arg10] : memref<8x8xf32, 3>
                %31 = affine.load %10[%arg8 mod 2, %arg9] : memref<2x8xf32, 3>
                %32 = affine.load %11[%arg8 mod 2, %arg10] : memref<2x8xf32, 3>
                %33 = arith.mulf %31, %32 : f32
                %34 = arith.addf %33, %30 : f32
                affine.store %34, %12[%arg9, %arg10] : memref<8x8xf32, 3>
              } {affine.loop = "unroll"}
            } {affine.loop = "unroll"}
          } {affine.loop = "unroll"}

          affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg7) {
            affine.for %arg8 = 0 to 4 {
              %27 = affine.vector_load %8[%c0 * 4 + %arg8] : memref<4xf32, 3>, vector<1xf32>
              affine.vector_store %27, %4[(%arg7 floordiv 8 + 1) mod 2, (%arg6 mod 2) * 4 + %arg8, %arg5 * 8 + %arg6 floordiv 2 + %c0 * 128] : memref<2x8x128xf32, 2>, vector<1xf32>
            } {affine.loop = "unroll"}
            %26 = affine.vector_load %9[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
            affine.vector_store %26, %5[(%arg7 floordiv 8 + 1) mod 2, (%arg5 * 16 + %arg6) floordiv 32 + %c0 * 8, ((%arg5 * 16 + %arg6) mod 32) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
            gpu.barrier
          }

          affine.for %arg8 = 0 to 8 {
            affine.for %arg9 = 0 to 8 {
              %26 = affine.load %12[%arg8, %arg9] : memref<8x8xf32, 3>
              %27 = affine.load %10[%c7 mod 2, %arg8] : memref<2x8xf32, 3>
              %28 = affine.load %11[%c7 mod 2, %arg9] : memref<2x8xf32, 3>
              %29 = arith.mulf %27, %28 : f32
              %30 = arith.addf %29, %26 : f32
              affine.store %30, %12[%arg8, %arg9] : memref<8x8xf32, 3>
            } {affine.loop = "unroll"}
          } {affine.loop = "unroll"}

          %22 = affine.vector_load %5[(%arg7 floordiv 8 + 1) mod 2, %c0_2, (%arg6 mod 4 + (((%arg5 * 16 + %arg6) floordiv 32) mod 4 + %c0 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %22, %11[0, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          %23 = affine.vector_load %5[(%arg7 floordiv 8 + 1) mod 2, %c0_2, (%arg6 mod 4 + (((%arg5 * 16 + %arg6) floordiv 32) mod 4 + %c1 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %23, %11[0, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          %24 = affine.vector_load %4[(%arg7 floordiv 8 + 1) mod 2, %c0_1, (((%arg5 * 16 + %arg6) mod 32) floordiv 4 + (((%arg5 * 16 + %arg6) floordiv 32) floordiv 4 + %c0 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %24, %10[0, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          %25 = affine.vector_load %4[(%arg7 floordiv 8 + 1) mod 2, %c0_1, (((%arg5 * 16 + %arg6) mod 32) floordiv 4 + (((%arg5 * 16 + %arg6) floordiv 32) floordiv 4 + %c1 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %25, %10[0, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        }

        affine.for %arg7 = 0 to 4 {
          %22 = affine.vector_load %12[%c0 + %arg7, %c0 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
          affine.vector_store %22, %12[%arg3 * 128 + (((%arg5 * 16 + %arg6) mod 32) floordiv 4 + (((%arg5 * 16 + %arg6) floordiv 32) floordiv 4 + (%c0 floordiv 4) * 2) * 8) * 4 + %arg7, %arg4 * 128 + (%arg6 mod 4 + (((%arg5 * 16 + %arg6) floordiv 32) mod 4 + (%c0 floordiv 4) * 4) * 4) * 4 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
        } {affine.loop = "unroll"}
        affine.for %arg7 = 0 to 4 {
          %22 = affine.vector_load %12[%c0 + %arg7, %c4 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
          affine.vector_store %22, %12[%arg3 * 128 + (((%arg5 * 16 + %arg6) mod 32) floordiv 4 + (((%arg5 * 16 + %arg6) floordiv 32) floordiv 4 + (%c0 floordiv 4) * 2) * 8) * 4 + %arg7, %arg4 * 128 + (%arg6 mod 4 + (((%arg5 * 16 + %arg6) floordiv 32) mod 4 + (%c4 floordiv 4) * 4) * 4) * 4 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
        } {affine.loop = "unroll"}
        affine.for %arg7 = 0 to 4 {
          %22 = affine.vector_load %12[%c4 + %arg7, %c0 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
          affine.vector_store %22, %12[%arg3 * 128 + (((%arg5 * 16 + %arg6) mod 32) floordiv 4 + (((%arg5 * 16 + %arg6) floordiv 32) floordiv 4 + (%c4 floordiv 4) * 2) * 8) * 4 + %arg7, %arg4 * 128 + (%arg6 mod 4 + (((%arg5 * 16 + %arg6) floordiv 32) mod 4 + (%c0 floordiv 4) * 4) * 4) * 4 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
        } {affine.loop = "unroll"}
        affine.for %arg7 = 0 to 4 {
          %22 = affine.vector_load %12[%c4 + %arg7, %c4 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
          affine.vector_store %22, %12[%arg3 * 128 + (((%arg5 * 16 + %arg6) mod 32) floordiv 4 + (((%arg5 * 16 + %arg6) floordiv 32) floordiv 4 + (%c4 floordiv 4) * 2) * 8) * 4 + %arg7, %arg4 * 128 + (%arg6 mod 4 + (((%arg5 * 16 + %arg6) floordiv 32) mod 4 + (%c4 floordiv 4) * 4) * 4) * 4 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
        } {affine.loop = "unroll"}
      }
    }
    return %arg2 : memref<4096x2048xf32, 1>
  }

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