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
module @demo {
  func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    %2 = memref.alloc() : memref<16x2048x64xf32, 1>
    affine.for %arg1 = 0 to 16 {
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      %3 = affine.for %arg2 = 0 to 2048 iter_args(%arg3 = %cst_1) -> (f32) {
        %9 = affine.for %arg4 = 0 to 64 iter_args(%arg5 = %arg3) -> (f32) {
          %10 = affine.load %arg0[%arg1, %arg2, %arg4] : memref<16x2048x64xf32, 1>
          %11 = arith.addf %10, %arg5 : f32
          affine.yield %11 : f32
        }
        affine.yield %9 : f32
      }
      %4 = arith.divf %3, %cst : f32
      %5 = affine.for %arg2 = 0 to 2048 iter_args(%arg3 = %cst_1) -> (f32) {
        %9 = affine.for %arg4 = 0 to 64 iter_args(%arg5 = %arg3) -> (f32) {
          %10 = affine.load %arg0[%arg1, %arg2, %arg4] : memref<16x2048x64xf32, 1>
          %11 = arith.subf %10, %4 : f32
          affine.store %11, %2[%arg1, %arg2, %arg4] : memref<16x2048x64xf32, 1>
          %12 = arith.mulf %11, %11 : f32
          %13 = arith.addf %12, %arg5 : f32
          affine.yield %13 : f32
        }
        affine.yield %9 : f32
      }
      %6 = arith.divf %5, %cst : f32
      %7 = arith.addf %6, %cst_0 : f32
      %8 = math.sqrt %7 : f32
      affine.for %arg2 = 0 to 2048 {
        affine.for %arg3 = 0 to 64 {
          %9 = affine.load %2[%arg1, %arg2, %arg3] : memref<16x2048x64xf32, 1>
          %10 = arith.divf %9, %8 : f32
          affine.store %10, %2[%arg1, %arg2, %arg3] : memref<16x2048x64xf32, 1>
        }
      }
    }
    return %2 : memref<16x2048x64xf32, 1>
  }
  %0 = memref.alloc() : memref<16x2048x64xf32, 1>
  %1 = func.call @LayerNorm_16_2048_64_axes_1_2(%0) : (memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1>
}
module @demo {
  func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    %2 = memref.alloc() : memref<16x2048x64xf32, 1>
    affine.for %arg1 = 0 to 16 {
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      %3 = affine.for %arg2 = 0 to 131072 iter_args(%arg3 = %cst_1) -> (f32) {
        %9 = affine.load %arg0[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %10 = arith.addf %9, %arg3 : f32
        affine.yield %10 : f32
      }
      %4 = arith.divf %3, %cst : f32
      %5 = affine.for %arg2 = 0 to 131072 iter_args(%arg3 = %cst_1) -> (f32) {
        %9 = affine.load %arg0[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %10 = arith.subf %9, %4 : f32
        affine.store %10, %2[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %11 = arith.mulf %10, %10 : f32
        %12 = arith.addf %11, %arg3 : f32
        affine.yield %12 : f32
      }
      %6 = arith.divf %5, %cst : f32
      %7 = arith.addf %6, %cst_0 : f32
      %8 = math.sqrt %7 : f32
      affine.for %arg2 = 0 to 131072 {
        %9 = affine.load %2[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %10 = arith.divf %9, %8 : f32
        affine.store %10, %2[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
      }
    }
    return %2 : memref<16x2048x64xf32, 1>
  }
  %0 = memref.alloc() : memref<16x2048x64xf32, 1>
  %1 = func.call @LayerNorm_16_2048_64_axes_1_2(%0) : (memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1>
}
module @demo {
  func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    %2 = memref.alloc() : memref<16x2048x64xf32, 1>
    affine.for %arg1 = 0 to 16 {
      %3 = memref.alloc() : memref<1xf32, 2>
      %4 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.store %cst_1, %3[%c0] : memref<1xf32, 2>
      affine.store %cst_1, %4[%c0] : memref<1xf32, 2>
      affine.for %arg2 = 0 to 131072 {
        %11 = affine.load %4[%c0] : memref<1xf32, 2>
        %12 = affine.load %arg0[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %13 = arith.addf %12, %11 : f32
        affine.store %13, %4[%c0] : memref<1xf32, 2>
      }
      %5 = affine.load %4[%c0] : memref<1xf32, 2>
      %6 = arith.divf %5, %cst : f32
      affine.for %arg2 = 0 to 131072 {
        %11 = affine.load %3[%c0] : memref<1xf32, 2>
        %12 = affine.load %arg0[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %13 = arith.subf %12, %6 : f32
        affine.store %13, %2[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %14 = arith.mulf %13, %13 : f32
        %15 = arith.addf %14, %11 : f32
        affine.store %15, %3[%c0] : memref<1xf32, 2>
      }
      %7 = affine.load %3[%c0] : memref<1xf32, 2>
      %8 = arith.divf %7, %cst : f32
      %9 = arith.addf %8, %cst_0 : f32
      %10 = math.sqrt %9 : f32
      affine.for %arg2 = 0 to 131072 {
        %11 = affine.load %2[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
        %12 = arith.divf %11, %10 : f32
        affine.store %12, %2[%arg1, %arg2 floordiv 64, %arg2 mod 64] : memref<16x2048x64xf32, 1>
      }
    }
    return %2 : memref<16x2048x64xf32, 1>
  }
  %0 = memref.alloc() : memref<16x2048x64xf32, 1>
  %1 = func.call @LayerNorm_16_2048_64_axes_1_2(%0) : (memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1>
}
module @demo {
  func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    %2 = memref.alloc() : memref<16x2048x64xf32, 1>
    affine.for %arg1 = 0 to 16 {
      %3 = memref.alloc() : memref<1xf32, 2>
      %4 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.store %cst_1, %3[%c0] : memref<1xf32, 2>
      affine.store %cst_1, %4[%c0] : memref<1xf32, 2>
      affine.for %arg2 = 0 to 131072 step 2048 {
        affine.for %arg3 = 0 to 2048 step 4 {
          affine.for %arg4 = 0 to 4 {
            %11 = affine.load %4[%c0] : memref<1xf32, 2>
            %12 = affine.load %arg0[%arg1, (%arg2 + %arg3 + %arg4) floordiv 64, (%arg2 + %arg3 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %13 = arith.addf %12, %11 : f32
            affine.store %13, %4[%c0] : memref<1xf32, 2>
          }
        }
      }
      %5 = affine.load %4[%c0] : memref<1xf32, 2>
      %6 = arith.divf %5, %cst : f32
      affine.for %arg2 = 0 to 131072 step 2048 {
        affine.for %arg3 = 0 to 2048 step 4 {
          affine.for %arg4 = 0 to 4 {
            %11 = affine.load %3[%c0] : memref<1xf32, 2>
            %12 = affine.load %arg0[%arg1, (%arg2 + %arg3 + %arg4) floordiv 64, (%arg2 + %arg3 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %13 = arith.subf %12, %6 : f32
            affine.store %13, %2[%arg1, (%arg2 + %arg3 + %arg4) floordiv 64, (%arg2 + %arg3 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %14 = arith.mulf %13, %13 : f32
            %15 = arith.addf %14, %11 : f32
            affine.store %15, %3[%c0] : memref<1xf32, 2>
          }
        }
      }
      %7 = affine.load %3[%c0] : memref<1xf32, 2>
      %8 = arith.divf %7, %cst : f32
      %9 = arith.addf %8, %cst_0 : f32
      %10 = math.sqrt %9 : f32
      affine.for %arg2 = 0 to 131072 step 2048 {
        affine.for %arg3 = 0 to 2048 step 4 {
          affine.for %arg4 = 0 to 4 {
            %11 = affine.load %2[%arg1, (%arg2 + %arg3 + %arg4) floordiv 64, (%arg2 + %arg3 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %12 = arith.divf %11, %10 : f32
            affine.store %12, %2[%arg1, (%arg2 + %arg3 + %arg4) floordiv 64, (%arg2 + %arg3 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
          }
        }
      }
    }
    return %2 : memref<16x2048x64xf32, 1>
  }
  %0 = memref.alloc() : memref<16x2048x64xf32, 1>
  %1 = func.call @LayerNorm_16_2048_64_axes_1_2(%0) : (memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1>
}
module @demo {
  func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    %2 = memref.alloc() : memref<16x2048x64xf32, 1>
    affine.for %arg1 = 0 to 16 {
      %3 = memref.alloc() : memref<1xf32, 2>
      %4 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.store %cst_1, %3[%c0] : memref<1xf32, 2>
      affine.store %cst_1, %4[%c0] : memref<1xf32, 2>
      affine.for %arg2 = 0 to 2048 step 4 {
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %11 = affine.load %4[%c0] : memref<1xf32, 2>
            %12 = affine.load %arg0[%arg1, (%arg3 + %arg2 + %arg4) floordiv 64, (%arg3 + %arg2 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %13 = arith.addf %12, %11 : f32
            affine.store %13, %4[%c0] : memref<1xf32, 2>
          }
        }
      }
      %5 = affine.load %4[%c0] : memref<1xf32, 2>
      %6 = arith.divf %5, %cst : f32
      affine.for %arg2 = 0 to 2048 step 4 {
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %11 = affine.load %3[%c0] : memref<1xf32, 2>
            %12 = affine.load %arg0[%arg1, (%arg3 + %arg2 + %arg4) floordiv 64, (%arg3 + %arg2 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %13 = arith.subf %12, %6 : f32
            affine.store %13, %2[%arg1, (%arg3 + %arg2 + %arg4) floordiv 64, (%arg3 + %arg2 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %14 = arith.mulf %13, %13 : f32
            %15 = arith.addf %14, %11 : f32
            affine.store %15, %3[%c0] : memref<1xf32, 2>
          }
        }
      }
      %7 = affine.load %3[%c0] : memref<1xf32, 2>
      %8 = arith.divf %7, %cst : f32
      %9 = arith.addf %8, %cst_0 : f32
      %10 = math.sqrt %9 : f32
      affine.for %arg2 = 0 to 2048 step 4 {
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %11 = affine.load %2[%arg1, (%arg3 + %arg2 + %arg4) floordiv 64, (%arg3 + %arg2 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %12 = arith.divf %11, %10 : f32
            affine.store %12, %2[%arg1, (%arg3 + %arg2 + %arg4) floordiv 64, (%arg3 + %arg2 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
          }
        }
      }
    }
    return %2 : memref<16x2048x64xf32, 1>
  }
  %0 = memref.alloc() : memref<16x2048x64xf32, 1>
  %1 = func.call @LayerNorm_16_2048_64_axes_1_2(%0) : (memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1>
}
module @demo {
  func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    %2 = memref.alloc() : memref<16x2048x64xf32, 1>
    affine.parallel (%arg1) = (0) to (16) {
      %3 = memref.alloc() : memref<1xf32, 2>
      %4 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.store %cst_1, %3[%c0] : memref<1xf32, 2>
      affine.store %cst_1, %4[%c0] : memref<1xf32, 2>
      affine.parallel (%arg2) = (0) to (512) {
        %11 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %12 = affine.load %4[%c0] : memref<1xf32, 2>
            %13 = affine.load %arg0[%arg1, (%arg3 + %11 + %arg4) floordiv 64, (%arg3 + %11 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %14 = arith.addf %13, %12 : f32
            affine.store %14, %4[%c0] : memref<1xf32, 2>
          }
        }
      }
      %5 = affine.load %4[%c0] : memref<1xf32, 2>
      %6 = arith.divf %5, %cst : f32
      affine.parallel (%arg2) = (0) to (512) {
        %11 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %12 = affine.load %3[%c0] : memref<1xf32, 2>
            %13 = affine.load %arg0[%arg1, (%arg3 + %11 + %arg4) floordiv 64, (%arg3 + %11 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %14 = arith.subf %13, %6 : f32
            affine.store %14, %2[%arg1, (%arg3 + %11 + %arg4) floordiv 64, (%arg3 + %11 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %15 = arith.mulf %14, %14 : f32
            %16 = arith.addf %15, %12 : f32
            affine.store %16, %3[%c0] : memref<1xf32, 2>
          }
        }
      }
      %7 = affine.load %3[%c0] : memref<1xf32, 2>
      %8 = arith.divf %7, %cst : f32
      %9 = arith.addf %8, %cst_0 : f32
      %10 = math.sqrt %9 : f32
      affine.parallel (%arg2) = (0) to (512) {
        %11 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %12 = affine.load %2[%arg1, (%arg3 + %11 + %arg4) floordiv 64, (%arg3 + %11 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %13 = arith.divf %12, %10 : f32
            affine.store %13, %2[%arg1, (%arg3 + %11 + %arg4) floordiv 64, (%arg3 + %11 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
          }
        }
      }
    }
    return %2 : memref<16x2048x64xf32, 1>
  }
  %0 = memref.alloc() : memref<16x2048x64xf32, 1>
  %1 = func.call @LayerNorm_16_2048_64_axes_1_2(%0) : (memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1>
}
module @demo {
  func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    %2 = memref.alloc() : memref<16x2048x64xf32, 1>
    affine.parallel (%arg1) = (0) to (16) {
      %3 = memref.alloc() : memref<1xf32, 2>
      %4 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.store %cst_1, %3[%c0] : memref<1xf32, 2>
      affine.store %cst_1, %4[%c0] : memref<1xf32, 2>
      affine.parallel (%arg2) = (0) to (512) {
        %11 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %12 = affine.load %4[%c0] : memref<1xf32, 2>
            %13 = affine.load %arg0[%arg1, (%arg3 + %11 + %arg4) floordiv 64, (%arg3 + %11 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %14 = arith.addf %13, %12 : f32
            affine.store %14, %4[%c0] : memref<1xf32, 2>
          }
        }
      }
      %5 = affine.load %4[%c0] : memref<1xf32, 2>
      %6 = arith.divf %5, %cst : f32
      affine.store %6, %4[%c0] : memref<1xf32, 2>
      affine.parallel (%arg2) = (0) to (512) {
        %11 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %12 = affine.load %3[%c0] : memref<1xf32, 2>
            %13 = affine.load %arg0[%arg1, (%arg3 + %11 + %arg4) floordiv 64, (%arg3 + %11 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %14 = affine.load %4[%c0] : memref<1xf32, 2>
            %15 = arith.subf %13, %14 : f32
            affine.store %15, %2[%arg1, (%arg3 + %11 + %arg4) floordiv 64, (%arg3 + %11 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %16 = arith.mulf %15, %15 : f32
            %17 = arith.addf %16, %12 : f32
            affine.store %17, %3[%c0] : memref<1xf32, 2>
          }
        }
      }
      %7 = affine.load %3[%c0] : memref<1xf32, 2>
      %8 = arith.divf %7, %cst : f32
      %9 = arith.addf %8, %cst_0 : f32
      %10 = math.sqrt %9 : f32
      affine.store %10, %3[%c0] : memref<1xf32, 2>
      affine.parallel (%arg2) = (0) to (512) {
        %11 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %12 = affine.load %2[%arg1, (%arg3 + %11 + %arg4) floordiv 64, (%arg3 + %11 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %13 = affine.load %3[%c0] : memref<1xf32, 2>
            %14 = arith.divf %12, %13 : f32
            affine.store %14, %2[%arg1, (%arg3 + %11 + %arg4) floordiv 64, (%arg3 + %11 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
          }
        }
      }
    }
    return %2 : memref<16x2048x64xf32, 1>
  }
  %0 = memref.alloc() : memref<16x2048x64xf32, 1>
  %1 = func.call @LayerNorm_16_2048_64_axes_1_2(%0) : (memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1>
}
module @demo {
  func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    %2 = memref.alloc() : memref<16x2048x64xf32, 1>
    affine.parallel (%arg1) = (0) to (16) {
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
          affine.for %arg4 = 0 to 4 {
            %6 = affine.load %4[%c0] : memref<1xf32, 2>
            %7 = affine.load %arg0[%arg1, (%arg3 + %5 + %arg4) floordiv 64, (%arg3 + %5 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %8 = arith.addf %7, %6 : f32
            affine.store %8, %4[%c0] : memref<1xf32, 2>
          }
        }
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %6 = affine.load %4[%c0] : memref<1xf32, 2>
          %7 = arith.divf %6, %cst : f32
          affine.store %7, %4[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %6 = affine.load %3[%c0] : memref<1xf32, 2>
            %7 = affine.load %arg0[%arg1, (%arg3 + %5 + %arg4) floordiv 64, (%arg3 + %5 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %8 = affine.load %4[%c0] : memref<1xf32, 2>
            %9 = arith.subf %7, %8 : f32
            affine.store %9, %2[%arg1, (%arg3 + %5 + %arg4) floordiv 64, (%arg3 + %5 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %10 = arith.mulf %9, %9 : f32
            %11 = arith.addf %10, %6 : f32
            affine.store %11, %3[%c0] : memref<1xf32, 2>
          }
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
          affine.for %arg4 = 0 to 4 {
            %6 = affine.load %2[%arg1, (%arg3 + %5 + %arg4) floordiv 64, (%arg3 + %5 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %7 = affine.load %3[%c0] : memref<1xf32, 2>
            %8 = arith.divf %6, %7 : f32
            affine.store %8, %2[%arg1, (%arg3 + %5 + %arg4) floordiv 64, (%arg3 + %5 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
          }
        }
      }
    }
    return %2 : memref<16x2048x64xf32, 1>
  }
  %0 = memref.alloc() : memref<16x2048x64xf32, 1>
  %1 = func.call @LayerNorm_16_2048_64_axes_1_2(%0) : (memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1>
}
module @demo {
  func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    %2 = memref.alloc() : memref<16x2048x64xf32, 1>
    affine.parallel (%arg1) = (0) to (16) {
      %3 = memref.alloc() : memref<2048xf32, 2>
      %4 = memref.alloc() : memref<1xf32, 2>
      %5 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.store %cst_1, %4[%c0] : memref<1xf32, 2>
      affine.store %cst_1, %5[%c0] : memref<1xf32, 2>
      affine.parallel (%arg2) = (0) to (512) {
        %6 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %arg0[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            affine.store %7, %3[%6 + %arg4] : memref<2048xf32, 2>
          }
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %5[%c0] : memref<1xf32, 2>
            %8 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %9 = arith.addf %8, %7 : f32
            affine.store %9, %5[%c0] : memref<1xf32, 2>
          }
        }
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %7 = affine.load %5[%c0] : memref<1xf32, 2>
          %8 = arith.divf %7, %cst : f32
          affine.store %8, %5[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %arg0[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            affine.store %7, %3[%6 + %arg4] : memref<2048xf32, 2>
          }
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %4[%c0] : memref<1xf32, 2>
            %8 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %9 = affine.load %5[%c0] : memref<1xf32, 2>
            %10 = arith.subf %8, %9 : f32
            affine.store %10, %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %11 = arith.mulf %10, %10 : f32
            %12 = arith.addf %11, %7 : f32
            affine.store %12, %4[%c0] : memref<1xf32, 2>
          }
        }
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %7 = affine.load %4[%c0] : memref<1xf32, 2>
          %8 = arith.divf %7, %cst : f32
          %9 = arith.addf %8, %cst_0 : f32
          %10 = math.sqrt %9 : f32
          affine.store %10, %4[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            affine.store %7, %3[%6 + %arg4] : memref<2048xf32, 2>
          }
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %8 = affine.load %4[%c0] : memref<1xf32, 2>
            %9 = arith.divf %7, %8 : f32
            affine.store %9, %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
          }
        }
      }
    }
    return %2 : memref<16x2048x64xf32, 1>
  }
  %0 = memref.alloc() : memref<16x2048x64xf32, 1>
  %1 = func.call @LayerNorm_16_2048_64_axes_1_2(%0) : (memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1>
}
module @demo {
  func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    %2 = memref.alloc() : memref<16x2048x64xf32, 1>
    affine.parallel (%arg1) = (0) to (16) {
      %3 = memref.alloc() : memref<2048xf32, 2>
      %4 = memref.alloc() : memref<1xf32, 2>
      %5 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.store %cst_1, %4[%c0] : memref<1xf32, 2>
      affine.store %cst_1, %5[%c0] : memref<1xf32, 2>
      affine.parallel (%arg2) = (0) to (512) {
        %6 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 step 4 {
            %7 = affine.vector_load %arg0[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %7, %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %5[%c0] : memref<1xf32, 2>
            %8 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %9 = arith.addf %8, %7 : f32
            affine.store %9, %5[%c0] : memref<1xf32, 2>
          }
          gpu.barrier
        }
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %7 = affine.load %5[%c0] : memref<1xf32, 2>
          %8 = arith.divf %7, %cst : f32
          affine.store %8, %5[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 step 4 {
            %7 = affine.vector_load %arg0[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %7, %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %4[%c0] : memref<1xf32, 2>
            %8 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %9 = affine.load %5[%c0] : memref<1xf32, 2>
            %10 = arith.subf %8, %9 : f32
            affine.store %10, %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
            %11 = arith.mulf %10, %10 : f32
            %12 = arith.addf %11, %7 : f32
            affine.store %12, %4[%c0] : memref<1xf32, 2>
          }
          gpu.barrier
        }
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %7 = affine.load %4[%c0] : memref<1xf32, 2>
          %8 = arith.divf %7, %cst : f32
          %9 = arith.addf %8, %cst_0 : f32
          %10 = math.sqrt %9 : f32
          affine.store %10, %4[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 step 4 {
            %7 = affine.vector_load %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %7, %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %8 = affine.load %4[%c0] : memref<1xf32, 2>
            %9 = arith.divf %7, %8 : f32
            affine.store %9, %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
          }
          gpu.barrier
        }
      }
    }
    return %2 : memref<16x2048x64xf32, 1>
  }
  %0 = memref.alloc() : memref<16x2048x64xf32, 1>
  %1 = func.call @LayerNorm_16_2048_64_axes_1_2(%0) : (memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1>
}
module @demo {
  func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    %2 = memref.alloc() : memref<16x2048x64xf32, 1>
    affine.parallel (%arg1) = (0) to (16) {
      %3 = memref.alloc() : memref<2048xf32, 2>
      %4 = memref.alloc() : memref<1xf32, 2>
      %5 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.store %cst_1, %4[%c0] : memref<1xf32, 2>
      affine.store %cst_1, %5[%c0] : memref<1xf32, 2>
      affine.parallel (%arg2) = (0) to (512) {
        %6 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 step 4 {
            %7 = affine.vector_load %arg0[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %7, %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %5[%c0] : memref<1xf32, 2>
            %8 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %9 = arith.addf %8, %7 : f32
            affine.store %9, %5[%c0] : memref<1xf32, 2>
          }
          gpu.barrier
        }
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %7 = affine.load %5[%c0] : memref<1xf32, 2>
          %8 = arith.divf %7, %cst : f32
          affine.store %8, %5[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 step 4 {
            %7 = affine.vector_load %arg0[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %7, %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %8 = affine.load %5[%c0] : memref<1xf32, 2>
            %9 = arith.subf %7, %8 : f32
            affine.store %9, %3[%6 + %arg4] : memref<2048xf32, 2>
          }
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            affine.store %7, %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
          }
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %8 = arith.mulf %7, %7 : f32
            affine.store %8, %3[%6 + %arg4] : memref<2048xf32, 2>
          }
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %8 = affine.load %4[%c0] : memref<1xf32, 2>
            %9 = arith.addf %7, %8 : f32
            affine.store %9, %4[%c0] : memref<1xf32, 2>
          }
          gpu.barrier
        }
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %7 = affine.load %4[%c0] : memref<1xf32, 2>
          %8 = arith.divf %7, %cst : f32
          %9 = arith.addf %8, %cst_0 : f32
          %10 = math.sqrt %9 : f32
          affine.store %10, %4[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 step 4 {
            %7 = affine.vector_load %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %7, %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %8 = affine.load %4[%c0] : memref<1xf32, 2>
            %9 = arith.divf %7, %8 : f32
            affine.store %9, %3[%6 + %arg4] : memref<2048xf32, 2>
          }
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            affine.store %7, %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>
          }
          gpu.barrier
        }
      }
    }
    return %2 : memref<16x2048x64xf32, 1>
  }
  %0 = memref.alloc() : memref<16x2048x64xf32, 1>
  %1 = func.call @LayerNorm_16_2048_64_axes_1_2(%0) : (memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1>
}
module @demo {
  func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    %2 = memref.alloc() : memref<16x2048x64xf32, 1>
    affine.parallel (%arg1) = (0) to (16) {
      %3 = memref.alloc() : memref<2048xf32, 2>
      %4 = memref.alloc() : memref<1xf32, 2>
      %5 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.store %cst_1, %4[%c0] : memref<1xf32, 2>
      affine.store %cst_1, %5[%c0] : memref<1xf32, 2>
      affine.parallel (%arg2) = (0) to (512) {
        %6 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 step 4 {
            %7 = affine.vector_load %arg0[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %7, %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %5[%c0] : memref<1xf32, 2>
            %8 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %9 = arith.addf %8, %7 : f32
            affine.store %9, %5[%c0] : memref<1xf32, 2>
          }
          gpu.barrier
        }
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %7 = affine.load %5[%c0] : memref<1xf32, 2>
          %8 = arith.divf %7, %cst : f32
          affine.store %8, %5[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 step 4 {
            %7 = affine.vector_load %arg0[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %7, %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %8 = affine.load %5[%c0] : memref<1xf32, 2>
            %9 = arith.subf %7, %8 : f32
            affine.store %9, %3[%6 + %arg4] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 step 4 {
            %7 = affine.vector_load %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
            affine.vector_store %7, %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %8 = arith.mulf %7, %7 : f32
            affine.store %8, %3[%6 + %arg4] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %8 = affine.load %4[%c0] : memref<1xf32, 2>
            %9 = arith.addf %7, %8 : f32
            affine.store %9, %4[%c0] : memref<1xf32, 2>
          }
          gpu.barrier
        }
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %7 = affine.load %4[%c0] : memref<1xf32, 2>
          %8 = arith.divf %7, %cst : f32
          %9 = arith.addf %8, %cst_0 : f32
          %10 = math.sqrt %9 : f32
          affine.store %10, %4[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 step 4 {
            %7 = affine.vector_load %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %7, %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %7 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %8 = affine.load %4[%c0] : memref<1xf32, 2>
            %9 = arith.divf %7, %8 : f32
            affine.store %9, %3[%6 + %arg4] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 step 4 {
            %7 = affine.vector_load %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
            affine.vector_store %7, %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
          }
          gpu.barrier
        }
      }
    }
    return %2 : memref<16x2048x64xf32, 1>
  }
  %0 = memref.alloc() : memref<16x2048x64xf32, 1>
  %1 = func.call @LayerNorm_16_2048_64_axes_1_2(%0) : (memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1>
}
module @demo {
  func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    %2 = memref.alloc() : memref<16x2048x64xf32, 1>
    affine.parallel (%arg1) = (0) to (16) {
      %3 = memref.alloc() : memref<2048xf32, 2>
      %4 = memref.alloc() : memref<1xf32, 2>
      %5 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.parallel (%arg2) = (0) to (512) {
        %6 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        %7 = memref.alloc() : memref<1xf32, 3>
        affine.store %cst_1, %7[%c0] : memref<1xf32, 3>
        %8 = memref.alloc() : memref<1xf32, 3>
        affine.store %cst_1, %8[%c0] : memref<1xf32, 3>
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 step 4 {
            %16 = affine.vector_load %arg0[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %16, %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          %9 = affine.load %3[%arg2] : memref<2048xf32, 2>
          %10 = affine.load %3[%arg2 + 512] : memref<2048xf32, 2>
          %11 = affine.load %3[%arg2 + 1024] : memref<2048xf32, 2>
          %12 = affine.load %3[%arg2 + 1536] : memref<2048xf32, 2>
          %13 = arith.addf %9, %10 : f32
          %14 = arith.addf %11, %12 : f32
          %15 = arith.addf %13, %14 : f32
          affine.store %15, %3[%arg2] : memref<2048xf32, 2>
          gpu.barrier
          affine.if affine_set<(d0) : (-d0 + 255 >= 0)>(%arg2) {
            %16 = affine.load %3[%arg2] : memref<2048xf32, 2>
            %17 = affine.load %3[%arg2 + 256] : memref<2048xf32, 2>
            %18 = arith.addf %16, %17 : f32
            affine.store %18, %3[%arg2] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.if affine_set<(d0) : (-d0 + 127 >= 0)>(%arg2) {
            %16 = affine.load %3[%arg2] : memref<2048xf32, 2>
            %17 = affine.load %3[%arg2 + 128] : memref<2048xf32, 2>
            %18 = arith.addf %16, %17 : f32
            affine.store %18, %3[%arg2] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.if affine_set<(d0) : (-d0 + 63 >= 0)>(%arg2) {
            %16 = affine.load %3[%arg2] : memref<2048xf32, 2>
            %17 = affine.load %3[%arg2 + 64] : memref<2048xf32, 2>
            %18 = arith.addf %16, %17 : f32
            affine.store %18, %3[%arg2] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.if affine_set<(d0) : (-d0 + 31 >= 0)>(%arg2) {
            %16 = affine.load %3[%arg2] : memref<2048xf32, 2>
            %17 = affine.load %3[%arg2 + 32] : memref<2048xf32, 2>
            %18 = arith.addf %16, %17 : f32
            %c32_i32 = arith.constant 32 : i32
            %c16_i32 = arith.constant 16 : i32
            %result, %valid = gpu.shuffle  down %18, %c16_i32, %c32_i32 : f32
            %19 = arith.addf %result, %18 : f32
            %c32_i32_2 = arith.constant 32 : i32
            %c8_i32 = arith.constant 8 : i32
            %result_3, %valid_4 = gpu.shuffle  down %19, %c8_i32, %c32_i32_2 : f32
            %20 = arith.addf %result_3, %19 : f32
            %c32_i32_5 = arith.constant 32 : i32
            %c4_i32 = arith.constant 4 : i32
            %result_6, %valid_7 = gpu.shuffle  down %20, %c4_i32, %c32_i32_5 : f32
            %21 = arith.addf %result_6, %20 : f32
            %c32_i32_8 = arith.constant 32 : i32
            %c2_i32 = arith.constant 2 : i32
            %result_9, %valid_10 = gpu.shuffle  down %21, %c2_i32, %c32_i32_8 : f32
            %22 = arith.addf %result_9, %21 : f32
            %c32_i32_11 = arith.constant 32 : i32
            %c1_i32 = arith.constant 1 : i32
            %result_12, %valid_13 = gpu.shuffle  down %22, %c1_i32, %c32_i32_11 : f32
            %23 = arith.addf %result_12, %22 : f32
            affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
              %24 = affine.load %8[%c0] : memref<1xf32, 3>
              %25 = arith.addf %24, %23 : f32
              affine.store %25, %8[%c0] : memref<1xf32, 3>
            }
          }
          gpu.barrier
        }
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %9 = affine.load %8[%c0] : memref<1xf32, 3>
          %10 = arith.divf %9, %cst : f32
          affine.store %10, %5[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 step 4 {
            %16 = affine.vector_load %arg0[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %16, %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %16 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %17 = affine.load %5[%c0] : memref<1xf32, 2>
            %18 = arith.subf %16, %17 : f32
            affine.store %18, %3[%6 + %arg4] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 step 4 {
            %16 = affine.vector_load %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
            affine.vector_store %16, %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %16 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %17 = arith.mulf %16, %16 : f32
            affine.store %17, %3[%6 + %arg4] : memref<2048xf32, 2>
          }
          gpu.barrier
          %9 = affine.load %3[%arg2] : memref<2048xf32, 2>
          %10 = affine.load %3[%arg2 + 512] : memref<2048xf32, 2>
          %11 = affine.load %3[%arg2 + 1024] : memref<2048xf32, 2>
          %12 = affine.load %3[%arg2 + 1536] : memref<2048xf32, 2>
          %13 = arith.addf %9, %10 : f32
          %14 = arith.addf %11, %12 : f32
          %15 = arith.addf %13, %14 : f32
          affine.store %15, %3[%arg2] : memref<2048xf32, 2>
          gpu.barrier
          affine.if affine_set<(d0) : (-d0 + 255 >= 0)>(%arg2) {
            %16 = affine.load %3[%arg2] : memref<2048xf32, 2>
            %17 = affine.load %3[%arg2 + 256] : memref<2048xf32, 2>
            %18 = arith.addf %16, %17 : f32
            affine.store %18, %3[%arg2] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.if affine_set<(d0) : (-d0 + 127 >= 0)>(%arg2) {
            %16 = affine.load %3[%arg2] : memref<2048xf32, 2>
            %17 = affine.load %3[%arg2 + 128] : memref<2048xf32, 2>
            %18 = arith.addf %16, %17 : f32
            affine.store %18, %3[%arg2] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.if affine_set<(d0) : (-d0 + 63 >= 0)>(%arg2) {
            %16 = affine.load %3[%arg2] : memref<2048xf32, 2>
            %17 = affine.load %3[%arg2 + 64] : memref<2048xf32, 2>
            %18 = arith.addf %16, %17 : f32
            affine.store %18, %3[%arg2] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.if affine_set<(d0) : (-d0 + 31 >= 0)>(%arg2) {
            %16 = affine.load %3[%arg2] : memref<2048xf32, 2>
            %17 = affine.load %3[%arg2 + 32] : memref<2048xf32, 2>
            %18 = arith.addf %16, %17 : f32
            %c32_i32 = arith.constant 32 : i32
            %c16_i32 = arith.constant 16 : i32
            %result, %valid = gpu.shuffle  down %18, %c16_i32, %c32_i32 : f32
            %19 = arith.addf %result, %18 : f32
            %c32_i32_2 = arith.constant 32 : i32
            %c8_i32 = arith.constant 8 : i32
            %result_3, %valid_4 = gpu.shuffle  down %19, %c8_i32, %c32_i32_2 : f32
            %20 = arith.addf %result_3, %19 : f32
            %c32_i32_5 = arith.constant 32 : i32
            %c4_i32 = arith.constant 4 : i32
            %result_6, %valid_7 = gpu.shuffle  down %20, %c4_i32, %c32_i32_5 : f32
            %21 = arith.addf %result_6, %20 : f32
            %c32_i32_8 = arith.constant 32 : i32
            %c2_i32 = arith.constant 2 : i32
            %result_9, %valid_10 = gpu.shuffle  down %21, %c2_i32, %c32_i32_8 : f32
            %22 = arith.addf %result_9, %21 : f32
            %c32_i32_11 = arith.constant 32 : i32
            %c1_i32 = arith.constant 1 : i32
            %result_12, %valid_13 = gpu.shuffle  down %22, %c1_i32, %c32_i32_11 : f32
            %23 = arith.addf %result_12, %22 : f32
            affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
              %24 = affine.load %7[%c0] : memref<1xf32, 3>
              %25 = arith.addf %24, %23 : f32
              affine.store %25, %7[%c0] : memref<1xf32, 3>
            }
          }
          gpu.barrier
        }
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %9 = affine.load %7[%c0] : memref<1xf32, 3>
          %10 = arith.divf %9, %cst : f32
          %11 = arith.addf %10, %cst_0 : f32
          %12 = math.sqrt %11 : f32
          affine.store %12, %4[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 step 4 {
            %9 = affine.vector_load %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %9, %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 {
            %9 = affine.load %3[%6 + %arg4] : memref<2048xf32, 2>
            %10 = affine.load %4[%c0] : memref<1xf32, 2>
            %11 = arith.divf %9, %10 : f32
            affine.store %11, %3[%6 + %arg4] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.for %arg4 = 0 to 4 step 4 {
            %9 = affine.vector_load %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
            affine.vector_store %9, %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
          }
          gpu.barrier
        }
      }
    }
    return %2 : memref<16x2048x64xf32, 1>
  }
  %0 = memref.alloc() : memref<16x2048x64xf32, 1>
  %1 = func.call @LayerNorm_16_2048_64_axes_1_2(%0) : (memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1>
}
module @demo {
  func.func @LayerNorm_16_2048_64_axes_1_2(%arg0: memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1> attributes {func.state = "cpu"} {
    %2 = memref.alloc() : memref<16x2048x64xf32, 1>
    affine.parallel (%arg1) = (0) to (16) {
      %3 = memref.alloc() : memref<2048xf32, 2>
      %4 = memref.alloc() : memref<1xf32, 2>
      %5 = memref.alloc() : memref<1xf32, 2>
      %c0 = arith.constant 0 : index
      %cst = arith.constant 1.310720e+05 : f32
      %cst_0 = arith.constant 9.99999974E-6 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      affine.parallel (%arg2) = (0) to (512) {
        %6 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
        %7 = memref.alloc() : memref<1xf32, 3>
        affine.store %cst_1, %7[%c0] : memref<1xf32, 3>
        %8 = memref.alloc() : memref<1xf32, 3>
        affine.store %cst_1, %8[%c0] : memref<1xf32, 3>
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 step 4 {
            %16 = affine.vector_load %arg0[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %16, %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          %9 = affine.load %3[%arg2] : memref<2048xf32, 2>
          %10 = affine.load %3[%arg2 + 512] : memref<2048xf32, 2>
          %11 = affine.load %3[%arg2 + 1024] : memref<2048xf32, 2>
          %12 = affine.load %3[%arg2 + 1536] : memref<2048xf32, 2>
          %13 = arith.addf %9, %10 : f32
          %14 = arith.addf %11, %12 : f32
          %15 = arith.addf %13, %14 : f32
          affine.store %15, %3[%arg2] : memref<2048xf32, 2>
          gpu.barrier
          affine.if affine_set<(d0) : (-d0 + 255 >= 0)>(%arg2) {
            %16 = affine.load %3[%arg2] : memref<2048xf32, 2>
            %17 = affine.load %3[%arg2 + 256] : memref<2048xf32, 2>
            %18 = arith.addf %16, %17 : f32
            affine.store %18, %3[%arg2] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.if affine_set<(d0) : (-d0 + 127 >= 0)>(%arg2) {
            %16 = affine.load %3[%arg2] : memref<2048xf32, 2>
            %17 = affine.load %3[%arg2 + 128] : memref<2048xf32, 2>
            %18 = arith.addf %16, %17 : f32
            affine.store %18, %3[%arg2] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.if affine_set<(d0) : (-d0 + 63 >= 0)>(%arg2) {
            %16 = affine.load %3[%arg2] : memref<2048xf32, 2>
            %17 = affine.load %3[%arg2 + 64] : memref<2048xf32, 2>
            %18 = arith.addf %16, %17 : f32
            affine.store %18, %3[%arg2] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.if affine_set<(d0) : (-d0 + 31 >= 0)>(%arg2) {
            %16 = affine.load %3[%arg2] : memref<2048xf32, 2>
            %17 = affine.load %3[%arg2 + 32] : memref<2048xf32, 2>
            %18 = arith.addf %16, %17 : f32
            %c32_i32 = arith.constant 32 : i32
            %c16_i32 = arith.constant 16 : i32
            %result, %valid = gpu.shuffle  down %18, %c16_i32, %c32_i32 : f32
            %19 = arith.addf %result, %18 : f32
            %c32_i32_2 = arith.constant 32 : i32
            %c8_i32 = arith.constant 8 : i32
            %result_3, %valid_4 = gpu.shuffle  down %19, %c8_i32, %c32_i32_2 : f32
            %20 = arith.addf %result_3, %19 : f32
            %c32_i32_5 = arith.constant 32 : i32
            %c4_i32 = arith.constant 4 : i32
            %result_6, %valid_7 = gpu.shuffle  down %20, %c4_i32, %c32_i32_5 : f32
            %21 = arith.addf %result_6, %20 : f32
            %c32_i32_8 = arith.constant 32 : i32
            %c2_i32 = arith.constant 2 : i32
            %result_9, %valid_10 = gpu.shuffle  down %21, %c2_i32, %c32_i32_8 : f32
            %22 = arith.addf %result_9, %21 : f32
            %c32_i32_11 = arith.constant 32 : i32
            %c1_i32 = arith.constant 1 : i32
            %result_12, %valid_13 = gpu.shuffle  down %22, %c1_i32, %c32_i32_11 : f32
            %23 = arith.addf %result_12, %22 : f32
            affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
              %24 = affine.load %8[%c0] : memref<1xf32, 3>
              %25 = arith.addf %24, %23 : f32
              affine.store %25, %8[%c0] : memref<1xf32, 3>
            }
          }
          gpu.barrier
        }
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %9 = affine.load %8[%c0] : memref<1xf32, 3>
          %10 = arith.divf %9, %cst : f32
          affine.store %10, %5[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 step 4 {
            %33 = affine.vector_load %arg0[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %33, %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          %9 = affine.load %5[%c0] : memref<1xf32, 2>
          %10 = affine.load %3[%arg2] : memref<2048xf32, 2>
          %11 = arith.subf %10, %9 : f32
          affine.store %11, %3[%arg2] : memref<2048xf32, 2>
          %12 = affine.load %3[%arg2 + 512] : memref<2048xf32, 2>
          %13 = arith.subf %12, %9 : f32
          affine.store %13, %3[%arg2 + 512] : memref<2048xf32, 2>
          %14 = affine.load %3[%arg2 + 1024] : memref<2048xf32, 2>
          %15 = arith.subf %14, %9 : f32
          affine.store %15, %3[%arg2 + 1024] : memref<2048xf32, 2>
          %16 = affine.load %3[%arg2 + 1536] : memref<2048xf32, 2>
          %17 = arith.subf %16, %9 : f32
          affine.store %17, %3[%arg2 + 1536] : memref<2048xf32, 2>
          gpu.barrier
          affine.for %arg4 = 0 to 4 step 4 {
            %33 = affine.vector_load %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
            affine.vector_store %33, %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
          }
          gpu.barrier
          %18 = affine.load %3[%arg2] : memref<2048xf32, 2>
          %19 = arith.mulf %18, %18 : f32
          affine.store %19, %3[%arg2] : memref<2048xf32, 2>
          %20 = affine.load %3[%arg2 + 512] : memref<2048xf32, 2>
          %21 = arith.mulf %20, %20 : f32
          affine.store %21, %3[%arg2 + 512] : memref<2048xf32, 2>
          %22 = affine.load %3[%arg2 + 1024] : memref<2048xf32, 2>
          %23 = arith.mulf %22, %22 : f32
          affine.store %23, %3[%arg2 + 1024] : memref<2048xf32, 2>
          %24 = affine.load %3[%arg2 + 1536] : memref<2048xf32, 2>
          %25 = arith.mulf %24, %24 : f32
          affine.store %25, %3[%arg2 + 1536] : memref<2048xf32, 2>
          gpu.barrier
          %26 = affine.load %3[%arg2] : memref<2048xf32, 2>
          %27 = affine.load %3[%arg2 + 512] : memref<2048xf32, 2>
          %28 = affine.load %3[%arg2 + 1024] : memref<2048xf32, 2>
          %29 = affine.load %3[%arg2 + 1536] : memref<2048xf32, 2>
          %30 = arith.addf %26, %27 : f32
          %31 = arith.addf %28, %29 : f32
          %32 = arith.addf %30, %31 : f32
          affine.store %32, %3[%arg2] : memref<2048xf32, 2>
          gpu.barrier
          affine.if affine_set<(d0) : (-d0 + 255 >= 0)>(%arg2) {
            %33 = affine.load %3[%arg2] : memref<2048xf32, 2>
            %34 = affine.load %3[%arg2 + 256] : memref<2048xf32, 2>
            %35 = arith.addf %33, %34 : f32
            affine.store %35, %3[%arg2] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.if affine_set<(d0) : (-d0 + 127 >= 0)>(%arg2) {
            %33 = affine.load %3[%arg2] : memref<2048xf32, 2>
            %34 = affine.load %3[%arg2 + 128] : memref<2048xf32, 2>
            %35 = arith.addf %33, %34 : f32
            affine.store %35, %3[%arg2] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.if affine_set<(d0) : (-d0 + 63 >= 0)>(%arg2) {
            %33 = affine.load %3[%arg2] : memref<2048xf32, 2>
            %34 = affine.load %3[%arg2 + 64] : memref<2048xf32, 2>
            %35 = arith.addf %33, %34 : f32
            affine.store %35, %3[%arg2] : memref<2048xf32, 2>
          }
          gpu.barrier
          affine.if affine_set<(d0) : (-d0 + 31 >= 0)>(%arg2) {
            %33 = affine.load %3[%arg2] : memref<2048xf32, 2>
            %34 = affine.load %3[%arg2 + 32] : memref<2048xf32, 2>
            %35 = arith.addf %33, %34 : f32
            %c32_i32 = arith.constant 32 : i32
            %c16_i32 = arith.constant 16 : i32
            %result, %valid = gpu.shuffle  down %35, %c16_i32, %c32_i32 : f32
            %36 = arith.addf %result, %35 : f32
            %c32_i32_2 = arith.constant 32 : i32
            %c8_i32 = arith.constant 8 : i32
            %result_3, %valid_4 = gpu.shuffle  down %36, %c8_i32, %c32_i32_2 : f32
            %37 = arith.addf %result_3, %36 : f32
            %c32_i32_5 = arith.constant 32 : i32
            %c4_i32 = arith.constant 4 : i32
            %result_6, %valid_7 = gpu.shuffle  down %37, %c4_i32, %c32_i32_5 : f32
            %38 = arith.addf %result_6, %37 : f32
            %c32_i32_8 = arith.constant 32 : i32
            %c2_i32 = arith.constant 2 : i32
            %result_9, %valid_10 = gpu.shuffle  down %38, %c2_i32, %c32_i32_8 : f32
            %39 = arith.addf %result_9, %38 : f32
            %c32_i32_11 = arith.constant 32 : i32
            %c1_i32 = arith.constant 1 : i32
            %result_12, %valid_13 = gpu.shuffle  down %39, %c1_i32, %c32_i32_11 : f32
            %40 = arith.addf %result_12, %39 : f32
            affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
              %41 = affine.load %7[%c0] : memref<1xf32, 3>
              %42 = arith.addf %41, %40 : f32
              affine.store %42, %7[%c0] : memref<1xf32, 3>
            }
          }
          gpu.barrier
        }
        affine.if affine_set<(d0) : (d0 == 0)>(%arg2) {
          %9 = affine.load %7[%c0] : memref<1xf32, 3>
          %10 = arith.divf %9, %cst : f32
          %11 = arith.addf %10, %cst_0 : f32
          %12 = math.sqrt %11 : f32
          affine.store %12, %4[%c0] : memref<1xf32, 2>
        }
        gpu.barrier
        affine.for %arg3 = 0 to 131072 step 2048 {
          affine.for %arg4 = 0 to 4 step 4 {
            %18 = affine.vector_load %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
            affine.vector_store %18, %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
          }
          gpu.barrier
          %9 = affine.load %4[%c0] : memref<1xf32, 2>
          %10 = affine.load %3[%arg2] : memref<2048xf32, 2>
          %11 = arith.divf %10, %9 : f32
          affine.store %11, %3[%arg2] : memref<2048xf32, 2>
          %12 = affine.load %3[%arg2 + 512] : memref<2048xf32, 2>
          %13 = arith.divf %12, %9 : f32
          affine.store %13, %3[%arg2 + 512] : memref<2048xf32, 2>
          %14 = affine.load %3[%arg2 + 1024] : memref<2048xf32, 2>
          %15 = arith.divf %14, %9 : f32
          affine.store %15, %3[%arg2 + 1024] : memref<2048xf32, 2>
          %16 = affine.load %3[%arg2 + 1536] : memref<2048xf32, 2>
          %17 = arith.divf %16, %9 : f32
          affine.store %17, %3[%arg2 + 1536] : memref<2048xf32, 2>
          gpu.barrier
          affine.for %arg4 = 0 to 4 step 4 {
            %18 = affine.vector_load %3[%6 + %arg4] : memref<2048xf32, 2>, vector<4xf32>
            affine.vector_store %18, %2[%arg1, (%arg3 + %6 + %arg4) floordiv 64, (%arg3 + %6 + %arg4) mod 64] : memref<16x2048x64xf32, 1>, vector<4xf32>
          }
          gpu.barrier
        }
      }
    }
    return %2 : memref<16x2048x64xf32, 1>
  }
  %0 = memref.alloc() : memref<16x2048x64xf32, 1>
  %1 = func.call @LayerNorm_16_2048_64_axes_1_2(%0) : (memref<16x2048x64xf32, 1>) -> memref<16x2048x64xf32, 1>
}

```