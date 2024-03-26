```C++
// gelu
Primitive shape: (16, 2048, 64, ); New shape: (2048, 1024)
BlockSize: (16, 16); GridSize: (16, 32)
-- mine gelu cost is : 0.004006

// mlir
func.func @gelu_Elementwise_16_2048_64(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    affine.for %arg1 = 0 to 2048 {
      affine.for %arg2 = 0 to 1024 {
        %2 = affine.load %arg0[(%arg1 * 1024 + %arg2) floordiv 131072, ((%arg1 * 1024 + %arg2) mod 131072) floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %cst = arith.constant 1.000000e+00 : f32
        %cst_0 = arith.constant 5.000000e-01 : f32
        %cst_1 = arith.constant 7.978840e-01 : f32
        %cst_2 = arith.constant 4.471500e-02 : f32
        %3 = arith.mulf %2, %cst_0 : f32
        %4 = arith.mulf %2, %2 : f32
        %5 = arith.mulf %2, %4 : f32
        %6 = arith.mulf %cst_2, %5 : f32
        %7 = arith.addf %6, %2 : f32
        %8 = arith.mulf %cst_1, %7 : f32
        %9 = math.tanh %8 : f32
        %10 = arith.addf %9, %cst : f32
        %11 = arith.mulf %3, %10 : f32
        affine.store %11, %arg0[(%arg1 * 1024 + %arg2) floordiv 131072, ((%arg1 * 1024 + %arg2) mod 131072) floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
      }
    }
    return %arg0 : memref<16x2048x64xf32, 1>
  }
// last step (result)
func.func @gelu_Elementwise_16_2048_64(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    affine.parallel (%arg1, %arg2) = (0, 0) to (32, 16) {
      %2 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg1)
      %3 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg2)
      affine.parallel (%arg3, %arg4) = (0, 0) to (16, 16) {
        %c0 = arith.constant 0 : index
        %4 = memref.alloc() : memref<4xf32, 3>
        %cst = arith.constant 4.471500e-02 : f32
        %cst_0 = arith.constant 7.978840e-01 : f32
        %cst_1 = arith.constant 5.000000e-01 : f32
        %cst_2 = arith.constant 1.000000e+00 : f32
        %5 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg3)
        %6 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg4)
        affine.for %arg5 = 0 to 4 {
          %7 = affine.vector_load %arg0[((%2 + %5 + %arg5) * 1024 + %3 + %6 + %c0 * 4) floordiv 131072, (((%2 + %5 + %arg5) * 1024 + %3 + %6 + %c0 * 4) mod 131072) floordiv 64, (%3 + %6 + %c0 * 4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
          affine.vector_store %7, %4[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
          affine.for %arg6 = 0 to 4 {
            %9 = affine.load %4[%arg6] : memref<4xf32, 3>
            %10 = arith.mulf %9, %cst_1 : f32
            %11 = arith.mulf %9, %9 : f32
            %12 = arith.mulf %9, %11 : f32
            %13 = arith.mulf %cst, %12 : f32
            %14 = arith.addf %13, %9 : f32
            %15 = arith.mulf %cst_0, %14 : f32
            %16 = math.tanh %15 : f32
            %17 = arith.addf %16, %cst_2 : f32
            %18 = arith.mulf %10, %17 : f32
            affine.store %18, %4[%arg6] : memref<4xf32, 3>
          } {affine.loop = "unroll"}
          %8 = affine.vector_load %4[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
          affine.vector_store %8, %arg0[((%2 + %5 + %arg5) * 1024 + %3 + %6 + %c0 * 4) floordiv 131072, (((%2 + %5 + %arg5) * 1024 + %3 + %6 + %c0 * 4) mod 131072) floordiv 64, (%3 + %6 + %c0 * 4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
        } {affine.loop = "unroll"}
      }
    }
    return %arg0 : memref<16x2048x64xf32, 1>
  }

//layerNorm
GridSize: 16; BlockSize 256
SpiltSize: 1024
PerThreadElem: 4
isUnroll: no
-- mine layerNorm cost is : 0.046198

GridSize: 16; BlockSize 256
SpiltSize: 1024
PerThreadElem: 4
isUnroll: yes
-- mine layerNorm cost is : 0.043443

GridSize: 16; BlockSize 256
SpiltSize: 2048
PerThreadElem: 8
isUnroll: yes
-- mine layerNorm cost is : 0.033610

GridSize: 16; BlockSize 512
SpiltSize: 2048
PerThreadElem: 4
isUnroll: yes
-- mine layerNorm cost is : 0.029194


// mlir
func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    affine.for %arg1 = 0 to 16 {
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      %2 = affine.for %arg2 = 0 to 2048 iter_args(%arg3 = %cst_1) -> (f32) {
        %8 = affine.for %arg4 = 0 to 64 iter_args(%arg5 = %arg3) -> (f32) {
          %9 = affine.load %arg0[%arg1, %arg2, %arg4] : memref<16x2048x64xf32, 1>
          %10 = arith.addf %9, %arg5 : f32
          affine.yield %10 : f32
        }
        affine.yield %8 : f32
      }
      %3 = arith.divf %2, %cst : f32
      %4 = affine.for %arg2 = 0 to 2048 iter_args(%arg3 = %cst_1) -> (f32) {
        %8 = affine.for %arg4 = 0 to 64 iter_args(%arg5 = %arg3) -> (f32) {
          %9 = affine.load %arg0[%arg1, %arg2, %arg4] : memref<16x2048x64xf32, 1>
          %10 = arith.subf %9, %3 : f32
          affine.store %10, %arg0[%arg1, %arg2, %arg4] : memref<16x2048x64xf32, 1>
          %11 = arith.mulf %10, %10 : f32
          %12 = arith.addf %11, %arg5 : f32
          affine.yield %12 : f32
        }
        affine.yield %8 : f32
      }
      %5 = arith.divf %4, %cst : f32
      %6 = arith.addf %5, %cst_0 : f32
      %7 = math.sqrt %6 : f32
      affine.for %arg2 = 0 to 2048 {
        affine.for %arg3 = 0 to 64 {
          %8 = affine.load %arg0[%arg1, %arg2, %arg3] : memref<16x2048x64xf32, 1>
          %9 = arith.divf %8, %7 : f32
          affine.store %9, %arg0[%arg1, %arg2, %arg3] : memref<16x2048x64xf32, 1>
        }
      }
    }
    return %arg0 : memref<16x2048x64xf32, 1>
  }

// 合并内层循环
func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    affine.for %arg1 = 0 to 16 {
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      %2 = affine.for %arg2 = 0 to 131072 iter_args(%arg3 = %cst_1) -> (f32) {
        %8 = affine.load %arg0[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %9 = arith.addf %8, %arg3 : f32
        affine.yield %9 : f32
      }
      %3 = arith.divf %2, %cst : f32
      %4 = affine.for %arg2 = 0 to 131072 iter_args(%arg3 = %cst_1) -> (f32) {
        %8 = affine.load %arg0[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %9 = arith.subf %8, %3 : f32
        affine.store %9, %arg0[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %10 = arith.mulf %9, %9 : f32
        %11 = arith.addf %10, %arg3 : f32
        affine.yield %11 : f32
      }
      %5 = arith.divf %4, %cst : f32
      %6 = arith.addf %5, %cst_0 : f32
      %7 = math.sqrt %6 : f32
      affine.for %arg2 = 0 to 131072 {
        %8 = affine.load %arg0[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %9 = arith.divf %8, %7 : f32
        affine.store %9, %arg0[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
      }
    }
    return %arg0 : memref<16x2048x64xf32, 1>
  }

// storeOp代替迭代变量
func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    affine.for %arg1 = 0 to 16 {
      %2 = memref.alloc() : memref<1xf32, 2>
      %3 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.store %cst_1, %2[%c0] : memref<1xf32, 2>
      affine.store %cst_1, %3[%c0] : memref<1xf32, 2>
      affine.for %arg2 = 0 to 131072 {
        %10 = affine.load %3[%c0] : memref<1xf32, 2>
        %11 = affine.load %arg0[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %12 = arith.addf %11, %10 : f32
        affine.store %12, %3[%c0] : memref<1xf32, 2>
      } 
      %4 = affine.load %3[%c0] : memref<1xf32, 2>
      %5 = arith.divf %4, %cst : f32
      affine.for %arg2 = 0 to 131072 {
        %10 = affine.load %2[%c0] : memref<1xf32, 2>
        %11 = affine.load %arg0[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %12 = arith.subf %11, %5 : f32
        affine.store %12, %arg0[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %13 = arith.mulf %12, %12 : f32
        %14 = arith.addf %13, %10 : f32
        affine.store %14, %2[%c0] : memref<1xf32, 2>
      }
      %6 = affine.load %2[%c0] : memref<1xf32, 2>
      %7 = arith.divf %6, %cst : f32
      %8 = arith.addf %7, %cst_0 : f32
      %9 = math.sqrt %8 : f32
      affine.for %arg2 = 0 to 131072 {
        %10 = affine.load %arg0[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %11 = arith.divf %10, %9 : f32
        affine.store %11, %arg0[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
      }
    }
    return %arg0 : memref<16x2048x64xf32, 1>
  }

//内层循环切开
func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    affine.for %arg1 = 0 to 16 {
      %2 = memref.alloc() : memref<1xf32, 2>
      %3 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.store %cst_1, %2[%c0] : memref<1xf32, 2>
      affine.store %cst_1, %3[%c0] : memref<1xf32, 2>
      affine.for %arg2 = 0 to 131072 step 2048 {
        affine.for %arg3 = 0 to 2048 step 4 {
          affine.for %arg4 = 0 to 4 {
            %10 = affine.load %3[%c0] : memref<1xf32, 2>
            %11 = affine.load %arg0[%arg1, (%arg2 + %arg3 + %arg4) floordiv 64, (%arg2 + %arg3 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %12 = arith.addf %11, %10 : f32
            affine.store %12, %3[%c0] : memref<1xf32, 2>
          }
        }
      }
      %4 = affine.load %3[%c0] : memref<1xf32, 2>
      %5 = arith.divf %4, %cst : f32
      affine.for %arg2 = 0 to 131072 step 2048 {
        affine.for %arg3 = 0 to 2048 step 4 {
          affine.for %arg4 = 0 to 4 {
            %10 = affine.load %2[%c0] : memref<1xf32, 2>
            %11 = affine.load %arg0[%arg1, (%arg2 + %arg3 + %arg4) floordiv 64, (%arg2 + %arg3 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %12 = arith.subf %11, %5 : f32
            affine.store %12, %arg0[%arg1, (%arg2 + %arg3 + %arg4) floordiv 64, (%arg2 + %arg3 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %13 = arith.mulf %12, %12 : f32
            %14 = arith.addf %13, %10 : f32
            affine.store %14, %2[%c0] : memref<1xf32, 2>
          }
        }
      }
      %6 = affine.load %2[%c0] : memref<1xf32, 2>
      %7 = arith.divf %6, %cst : f32
      %8 = arith.addf %7, %cst_0 : f32
      %9 = math.sqrt %8 : f32
      affine.for %arg2 = 0 to 131072 step 2048 {
        affine.for %arg3 = 0 to 2048 step 4 {
          affine.for %arg4 = 0 to 4 {
            %10 = affine.load %arg0[%arg1, (%arg2 + %arg3 + %arg4) floordiv 64, (%arg2 + %arg3 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %11 = arith.divf %10, %9 : f32
            affine.store %11, %arg0[%arg1, (%arg2 + %arg3 + %arg4) floordiv 64, (%arg2 + %arg3 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
          }
        }
      }
    }
    return %arg0 : memref<16x2048x64xf32, 1>
  }

// 将并行循环，移动到最外层
  func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    affine.for %arg1 = 0 to 16 {
      %2 = memref.alloc() : memref<1xf32, 2>
      %3 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.store %cst_1, %2[%c0] : memref<1xf32, 2>
      affine.store %cst_1, %3[%c0] : memref<1xf32, 2>
      affine.for %arg2 = 0 to 2048 step 4 {
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %10 = affine.load %3[%c0] : memref<1xf32, 2>
            %11 = affine.load %arg0[%arg1, (%arg3 + %arg2 + %arg4) floordiv 64, (%arg3 + %arg2 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %12 = arith.addf %11, %10 : f32
            affine.store %12, %3[%c0] : memref<1xf32, 2>
          }
        }
      }
      %4 = affine.load %3[%c0] : memref<1xf32, 2>
      %5 = arith.divf %4, %cst : f32
      affine.for %arg2 = 0 to 2048 step 4 {
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %10 = affine.load %2[%c0] : memref<1xf32, 2>
            %11 = affine.load %arg0[%arg1, (%arg3 + %arg2 + %arg4) floordiv 64, (%arg3 + %arg2 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %12 = arith.subf %11, %5 : f32
            affine.store %12, %arg0[%arg1, (%arg3 + %arg2 + %arg4) floordiv 64, (%arg3 + %arg2 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %13 = arith.mulf %12, %12 : f32
            %14 = arith.addf %13, %10 : f32
            affine.store %14, %2[%c0] : memref<1xf32, 2>
          }
        }
      }
      %6 = affine.load %2[%c0] : memref<1xf32, 2>
      %7 = arith.divf %6, %cst : f32
      %8 = arith.addf %7, %cst_0 : f32
      %9 = math.sqrt %8 : f32
      affine.for %arg2 = 0 to 2048 step 4 {
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %10 = affine.load %arg0[%arg1, (%arg3 + %arg2 + %arg4) floordiv 64, (%arg3 + %arg2 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %11 = arith.divf %10, %9 : f32
            affine.store %11, %arg0[%arg1, (%arg3 + %arg2 + %arg4) floordiv 64, (%arg3 + %arg2 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
          }
        }
      }
    }
    return %arg0 : memref<16x2048x64xf32, 1>
  }

// 并行
func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    affine.parallel (%arg1) = (0) to (16) {
      %2 = memref.alloc() : memref<1xf32, 2>
      %3 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.store %cst_1, %2[%c0] : memref<1xf32, 2>
      affine.store %cst_1, %3[%c0] : memref<1xf32, 2>
      affine.parallel (%arg2) = (0) to (512) {
        %10 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %11 = affine.load %3[%c0] : memref<1xf32, 2>
            %12 = affine.load %arg0[%arg1, (%arg3 + %10 + %arg4) floordiv 64, (%arg3 + %10 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %13 = arith.addf %12, %11 : f32
            affine.store %13, %3[%c0] : memref<1xf32, 2>
          }
        }
      }
      %4 = affine.load %3[%c0] : memref<1xf32, 2>
      %5 = arith.divf %4, %cst : f32
      affine.parallel (%arg2) = (0) to (512) {
        %10 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %11 = affine.load %2[%c0] : memref<1xf32, 2>
            %12 = affine.load %arg0[%arg1, (%arg3 + %10 + %arg4) floordiv 64, (%arg3 + %10 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %13 = arith.subf %12, %5 : f32
            affine.store %13, %arg0[%arg1, (%arg3 + %10 + %arg4) floordiv 64, (%arg3 + %10 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %14 = arith.mulf %13, %13 : f32
            %15 = arith.addf %14, %11 : f32
            affine.store %15, %2[%c0] : memref<1xf32, 2>
          }
        }
      }
      %6 = affine.load %2[%c0] : memref<1xf32, 2>
      %7 = arith.divf %6, %cst : f32
      %8 = arith.addf %7, %cst_0 : f32
      %9 = math.sqrt %8 : f32
      affine.parallel (%arg2) = (0) to (512) {
        %10 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %11 = affine.load %arg0[%arg1, (%arg3 + %10 + %arg4) floordiv 64, (%arg3 + %10 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %12 = arith.divf %11, %9 : f32
            affine.store %12, %arg0[%arg1, (%arg3 + %10 + %arg4) floordiv 64, (%arg3 + %10 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
          }
        }
      }
    }
    return %arg0 : memref<16x2048x64xf32, 1>
  }

// 将并行移动到最外层，添加求mean的条件语句，以及线程同步
func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    affine.parallel (%arg1) = (0) to (16) {
      %2 = memref.alloc() : memref<1xf32, 2>
      %3 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.store %cst_1, %2[%c0] : memref<1xf32, 2>
      affine.store %cst_1, %3[%c0] : memref<1xf32, 2>
      affine.parallel (%arg2) = (0) to (512) {
        %4 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %5 = affine.load %3[%c0] : memref<1xf32, 2>
            %6 = affine.load %arg0[%arg1, (%arg3 + %4 + %arg4) floordiv 64, (%arg3 + %4 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %7 = arith.addf %6, %5 : f32
            affine.store %7, %3[%c0] : memref<1xf32, 2>
          }
        }
        gpu.barrier
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %5 = affine.load %3[%c0] : memref<1xf32, 2>
          %6 = arith.divf %5, %cst : f32
          affine.store %6, %3[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %5 = affine.load %2[%c0] : memref<1xf32, 2>
            %6 = affine.load %arg0[%arg1, (%arg3 + %4 + %arg4) floordiv 64, (%arg3 + %4 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %7 = affine.load %3[%c0] : memref<1xf32, 2>
            %8 = arith.subf %6, %7 : f32
            affine.store %8, %arg0[%arg1, (%arg3 + %4 + %arg4) floordiv 64, (%arg3 + %4 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %9 = arith.mulf %8, %8 : f32
            %10 = arith.addf %9, %5 : f32
            affine.store %10, %2[%c0] : memref<1xf32, 2>
          }
        }
        gpu.barrier
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %5 = affine.load %2[%c0] : memref<1xf32, 2>
          %6 = arith.divf %5, %cst : f32
          %7 = arith.addf %6, %cst_0 : f32
          %8 = math.sqrt %7 : f32
          affine.store %8, %2[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %5 = affine.load %arg0[%arg1, (%arg3 + %4 + %arg4) floordiv 64, (%arg3 + %4 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %6 = affine.load %2[%c0] : memref<1xf32, 2>
            %7 = arith.divf %5, %6 : f32
            affine.store %7, %arg0[%arg1, (%arg3 + %4 + %arg4) floordiv 64, (%arg3 + %4 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
          }
        }
      }
    }
    return %arg0 : memref<16x2048x64xf32, 1>
  }

// 设置2048的shared memory，将input向量化读入shared，并替换之前直接从input读取的op操作,同步线程（read、cache read）
func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    affine.parallel (%arg1) = (0) to (16) {
      %2 = memref.alloc() : memref<2048xf32, 2>
      %3 = memref.alloc() : memref<1xf32, 2>
      %4 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.store %cst_1, %3[%c0] : memref<1xf32, 2>
      affine.store %cst_1, %4[%c0] : memref<1xf32, 2>
      affine.parallel (%arg2) = (0) to (512) {
        %5 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 1 {
            %6 = affine.vector_load %arg0[%arg1, (%arg3 + %5 + %arg4 * 4) floordiv 64, (%arg3 + %5 + %arg4 * 4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %6, %2[%5 + %arg4 * 4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %6 = affine.load %4[%c0] : memref<1xf32, 2>
            %7 = affine.load %2[%5 + %arg4] : memref<2048xf32, 2>
            %8 = arith.addf %7, %6 : f32
            affine.store %8, %4[%c0] : memref<1xf32, 2>
          }
          gpu.barrier
        }
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %6 = affine.load %4[%c0] : memref<1xf32, 2>
          %7 = arith.divf %6, %cst : f32
          affine.store %7, %4[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 1 {
            %6 = affine.vector_load %arg0[%arg1, (%arg3 + %5 + %arg4 * 4) floordiv 64, (%arg3 + %5 + %arg4 * 4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %6, %2[%5 + %arg4 * 4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %6 = affine.load %3[%c0] : memref<1xf32, 2>
            %7 = affine.load %2[%5 + %arg4] : memref<2048xf32, 2>
            %8 = affine.load %4[%c0] : memref<1xf32, 2>
            %9 = arith.subf %7, %8 : f32
            affine.store %9, %arg0[%arg1, (%arg3 + %5 + %arg4) floordiv 64, (%arg3 + %5 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %10 = arith.mulf %9, %9 : f32
            %11 = arith.addf %10, %6 : f32
            affine.store %11, %3[%c0] : memref<1xf32, 2>
          }
          gpu.barrier
        }
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %6 = affine.load %3[%c0] : memref<1xf32, 2>
          %7 = arith.divf %6, %cst : f32
          %8 = arith.addf %7, %cst_0 : f32
          %9 = math.sqrt %8 : f32
          affine.store %9, %3[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 1 {
            %6 = affine.vector_load %arg0[%arg1, (%arg3 + %5 + %arg4 * 4) floordiv 64, (%arg3 + %5 + %arg4 * 4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %6, %2[%5 + %arg4 * 4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %6 = affine.load %2[%5 + %arg4] : memref<2048xf32, 2>
            %7 = affine.load %3[%c0] : memref<1xf32, 2>
            %8 = arith.divf %6, %7 : f32
            affine.store %8, %arg0[%arg1, (%arg3 + %5 + %arg4) floordiv 64, (%arg3 + %5 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
          }
          gpu.barrier
        }
      }
    }
    return %arg0 : memref<16x2048x64xf32, 1>
  }
```