```C++

  func.func @Matmul_m2048n2048k1024(%arg0: memref<2048x1024xf32, 1>, %arg1: memref<1024x2048xf32, 1>) -> memref<2048x2048xf32, 1> attributes {func.state = "gpu"} {
    %3 = memref.alloc() : memref<2048x2048xf32, 1>
    affine.parallel (%arg2, %arg3) = (0, 0) to (16, 16) {
      %4 = memref.alloc() : memref<2x8x128xf32, 2>
      %5 = memref.alloc() : memref<2x8x128xf32, 2>
      %6 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg2)
      %7 = affine.apply affine_map<(d0) -> (d0 * 128)>(%arg3)
      affine.parallel (%arg4, %arg5) = (0, 0) to (16, 16) {
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
        %13 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg4)
        %14 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg5)
        affine.for %arg6 = 0 to 8 {
          affine.for %arg7 = 0 to 8 {
            %cst = arith.constant 0.000000e+00 : f32
            affine.store %cst, %12[%arg6, %arg7] : memref<8x8xf32, 3>
          } {affine.loop = "unroll"}
        } {affine.loop = "unroll"}

        %15 = affine.vector_load %arg0[%arg4 * 8 + %arg5 floordiv 2 + %c0 * 128 + %arg2 * 128, (%arg5 mod 2) * 4 + %c0_3] : memref<2048x1024xf32, 1>, vector<4xf32>
        affine.vector_store %15, %8[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
        %16 = affine.vector_load %arg1[(%arg4 * 16 + %arg5) floordiv 32 + %c0 * 8 + %c0_4, ((%arg4 * 16 + %arg5) mod 32) * 4 + %arg3 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
        affine.vector_store %16, %9[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
        affine.for %arg6 = 0 to 4 {
          %22 = affine.vector_load %8[%c0 * 4 + %arg6] : memref<4xf32, 3>, vector<1xf32>
          affine.vector_store %22, %4[0, (%arg5 mod 2) * 4 + %arg6, %arg4 * 8 + %arg5 floordiv 2 + %c0 * 128] : memref<2x8x128xf32, 2>, vector<1xf32>
        } {affine.loop = "unroll"}
        %17 = affine.vector_load %9[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
        affine.vector_store %17, %5[0, (%arg4 * 16 + %arg5) floordiv 32 + %c0 * 8, ((%arg4 * 16 + %arg5) mod 32) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        gpu.barrier

        %18 = affine.vector_load %4[(%c0_0 floordiv 8) mod 2, %c0_1, (((%arg4 * 16 + %arg5) mod 32) floordiv 4 + (((%arg4 * 16 + %arg5) floordiv 32) floordiv 4 + %c0 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %18, %10[0, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        %19 = affine.vector_load %4[(%c0_0 floordiv 8) mod 2, %c0_1, (((%arg4 * 16 + %arg5) mod 32) floordiv 4 + (((%arg4 * 16 + %arg5) floordiv 32) floordiv 4 + %c1 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %19, %10[0, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        %20 = affine.vector_load %5[(%c0 floordiv 8) mod 2, %c0_2, (%arg5 mod 4 + (((%arg4 * 16 + %arg5) floordiv 32) mod 4 + %c0 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %20, %11[0, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        %21 = affine.vector_load %5[(%c0 floordiv 8) mod 2, %c0_2, (%arg5 mod 4 + (((%arg4 * 16 + %arg5) floordiv 32) mod 4 + %c1 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
        affine.vector_store %21, %11[0, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>

        affine.for %arg6 = 0 to 1024 step 8 {
          affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg6) {
            %26 = affine.vector_load %arg0[%arg4 * 8 + %arg5 floordiv 2 + %c0 * 128 + %arg2 * 128, (%arg5 mod 2) * 4 + %arg6 + 8] : memref<2048x1024xf32, 1>, vector<4xf32>
            affine.vector_store %26, %8[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
            %27 = affine.vector_load %arg1[(%arg4 * 16 + %arg5) floordiv 32 + %c0 * 8 + %arg6 + 8, ((%arg4 * 16 + %arg5) mod 32) * 4 + %arg3 * 128] : memref<1024x2048xf32, 1>, vector<4xf32>
            affine.vector_store %27, %9[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
          }
          affine.for %arg7 = 0 to 7 {
            %26 = affine.vector_load %4[(%arg6 floordiv 8) mod 2, %arg7 + 1, (((%arg4 * 16 + %arg5) mod 32) floordiv 4 + (((%arg4 * 16 + %arg5) floordiv 32) floordiv 4 + %c0 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
            affine.vector_store %26, %10[(%arg7 + 1) mod 2, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
            %27 = affine.vector_load %4[(%arg6 floordiv 8) mod 2, %arg7 + 1, (((%arg4 * 16 + %arg5) mod 32) floordiv 4 + (((%arg4 * 16 + %arg5) floordiv 32) floordiv 4 + %c1 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
            affine.vector_store %27, %10[(%arg7 + 1) mod 2, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
            %28 = affine.vector_load %5[(%arg6 floordiv 8) mod 2, %arg7 + 1, (%arg5 mod 4 + (((%arg4 * 16 + %arg5) floordiv 32) mod 4 + %c0 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
            affine.vector_store %28, %11[(%arg7 + 1) mod 2, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
            %29 = affine.vector_load %5[(%arg6 floordiv 8) mod 2, %arg7 + 1, (%arg5 mod 4 + (((%arg4 * 16 + %arg5) floordiv 32) mod 4 + %c1 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
            affine.vector_store %29, %11[(%arg7 + 1) mod 2, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>

            affine.for %arg8 = 0 to 8 {
              affine.for %arg9 = 0 to 8 {
                %30 = affine.load %12[%arg8, %arg9] : memref<8x8xf32, 3>
                %31 = affine.load %10[%arg7 mod 2, %arg8] : memref<2x8xf32, 3>
                %32 = affine.load %11[%arg7 mod 2, %arg9] : memref<2x8xf32, 3>
                %33 = arith.mulf %31, %32 : f32
                %34 = arith.addf %33, %30 : f32
                affine.store %34, %12[%arg8, %arg9] : memref<8x8xf32, 3>
              } {affine.loop = "unroll"}
            } {affine.loop = "unroll"}
          } {affine.loop = "unroll"}

          affine.if affine_set<(d0) : (-d0 + 1008 >= 0)>(%arg6) {
            affine.for %arg7 = 0 to 4 {
              %27 = affine.vector_load %8[%c0 * 4 + %arg7] : memref<4xf32, 3>, vector<1xf32>
              affine.vector_store %27, %4[(%arg6 floordiv 8 + 1) mod 2, (%arg5 mod 2) * 4 + %arg7, %arg4 * 8 + %arg5 floordiv 2 + %c0 * 128] : memref<2x8x128xf32, 2>, vector<1xf32>
            } {affine.loop = "unroll"}
            %26 = affine.vector_load %9[%c0 * 4] : memref<4xf32, 3>, vector<4xf32>
            affine.vector_store %26, %5[(%arg6 floordiv 8 + 1) mod 2, (%arg4 * 16 + %arg5) floordiv 32 + %c0 * 8, ((%arg4 * 16 + %arg5) mod 32) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
            gpu.barrier
          }
          affine.for %arg7 = 0 to 8 {
            affine.for %arg8 = 0 to 8 {
              %26 = affine.load %12[%arg7, %arg8] : memref<8x8xf32, 3>
              %27 = affine.load %10[%c7 mod 2, %arg7] : memref<2x8xf32, 3>
              %28 = affine.load %11[%c7 mod 2, %arg8] : memref<2x8xf32, 3>
              %29 = arith.mulf %27, %28 : f32
              %30 = arith.addf %29, %26 : f32
              affine.store %30, %12[%arg7, %arg8] : memref<8x8xf32, 3>
            } {affine.loop = "unroll"}
          } {affine.loop = "unroll"}

          %22 = affine.vector_load %5[(%arg6 floordiv 8 + 1) mod 2, %c0_2, (%arg5 mod 4 + (((%arg4 * 16 + %arg5) floordiv 32) mod 4 + %c0 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %22, %11[0, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          %23 = affine.vector_load %5[(%arg6 floordiv 8 + 1) mod 2, %c0_2, (%arg5 mod 4 + (((%arg4 * 16 + %arg5) floordiv 32) mod 4 + %c1 * 4) * 4) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %23, %11[0, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          %24 = affine.vector_load %4[(%arg6 floordiv 8 + 1) mod 2, %c0_1, (((%arg4 * 16 + %arg5) mod 32) floordiv 4 + (((%arg4 * 16 + %arg5) floordiv 32) floordiv 4 + %c0 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %24, %10[0, %c0 * 4] : memref<2x8xf32, 3>, vector<4xf32>
          %25 = affine.vector_load %4[(%arg6 floordiv 8 + 1) mod 2, %c0_1, (((%arg4 * 16 + %arg5) mod 32) floordiv 4 + (((%arg4 * 16 + %arg5) floordiv 32) floordiv 4 + %c1 * 2) * 8) * 4] : memref<2x8x128xf32, 2>, vector<4xf32>
          affine.vector_store %25, %10[0, %c1 * 4] : memref<2x8xf32, 3>, vector<4xf32>
        }
        affine.for %arg6 = 0 to 4 {
          %22 = affine.vector_load %12[%c0 + %arg6, %c0 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
          affine.vector_store %22, %3[%arg2 * 128 + (((%arg4 * 16 + %arg5) mod 32) floordiv 4 + (((%arg4 * 16 + %arg5) floordiv 32) floordiv 4 + (%c0 floordiv 4) * 2) * 8) * 4 + %arg6, %arg3 * 128 + (%arg5 mod 4 + (((%arg4 * 16 + %arg5) floordiv 32) mod 4 + (%c0 floordiv 4) * 4) * 4) * 4 + %c0] : memref<2048x2048xf32, 1>, vector<4xf32>
        } {affine.loop = "unroll"}
        affine.for %arg6 = 0 to 4 {
          %22 = affine.vector_load %12[%c0 + %arg6, %c4 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
          affine.vector_store %22, %3[%arg2 * 128 + (((%arg4 * 16 + %arg5) mod 32) floordiv 4 + (((%arg4 * 16 + %arg5) floordiv 32) floordiv 4 + (%c0 floordiv 4) * 2) * 8) * 4 + %arg6, %arg3 * 128 + (%arg5 mod 4 + (((%arg4 * 16 + %arg5) floordiv 32) mod 4 + (%c4 floordiv 4) * 4) * 4) * 4 + %c0] : memref<2048x2048xf32, 1>, vector<4xf32>
        } {affine.loop = "unroll"}
        affine.for %arg6 = 0 to 4 {
          %22 = affine.vector_load %12[%c4 + %arg6, %c0 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
          affine.vector_store %22, %3[%arg2 * 128 + (((%arg4 * 16 + %arg5) mod 32) floordiv 4 + (((%arg4 * 16 + %arg5) floordiv 32) floordiv 4 + (%c4 floordiv 4) * 2) * 8) * 4 + %arg6, %arg3 * 128 + (%arg5 mod 4 + (((%arg4 * 16 + %arg5) floordiv 32) mod 4 + (%c0 floordiv 4) * 4) * 4) * 4 + %c0] : memref<2048x2048xf32, 1>, vector<4xf32>
        } {affine.loop = "unroll"}
        affine.for %arg6 = 0 to 4 {
          %22 = affine.vector_load %12[%c4 + %arg6, %c4 + %c0] : memref<8x8xf32, 3>, vector<4xf32>
          affine.vector_store %22, %3[%arg2 * 128 + (((%arg4 * 16 + %arg5) mod 32) floordiv 4 + (((%arg4 * 16 + %arg5) floordiv 32) floordiv 4 + (%c4 floordiv 4) * 2) * 8) * 4 + %arg6, %arg3 * 128 + (%arg5 mod 4 + (((%arg4 * 16 + %arg5) floordiv 32) mod 4 + (%c4 floordiv 4) * 4) * 4) * 4 + %c0] : memref<2048x2048xf32, 1>, vector<4xf32>
        } {affine.loop = "unroll"}
      }
    }
    return %3 : memref<2048x2048xf32, 1>
  }



func.func @Fused_Multi_Head_Attention_8_32_SL2048_HD64(%arg0: memref<8x32x2048x64xf32, 1>, %arg1: memref<8x32x2048x64xf32, 1>, %arg2: memref<8x32x2048x64xf32, 1>, %arg3: memref<8x32x2048x64xf32, 1>) -> memref<8x32x2048x64xf32, 1> attributes {func.state = "gpu"} {
    affine.parallel (%arg4, %arg5, %arg6) = (0, 0, 0) to (8, 32, 16) {
      affine.parallel (%arg7) = (0) to (128) {
        %6 = memref.alloc() : memref<8x8xf32, 3>
        %7 = memref.alloc() : memref<128xf32, 2>
        %8 = memref.alloc() : memref<128xf32, 2>
        %9 = memref.alloc() : memref<128xf32, 2>
        %10 = memref.alloc() : memref<128x64xf32, 2>
        %11 = memref.alloc() : memref<8x64xf32, 2>
        %12 = memref.alloc() : memref<8x64xf32, 2>
        %13 = memref.alloc() : memref<8x128xf32, 2>
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant -3.40282347E+38 : f32
        affine.for %arg8 = 0 to 8 {
          affine.for %arg9 = 0 to 8 {
            affine.store %cst, %6[%arg8, %arg9] : memref<8x8xf32, 3>
          }
        }
        affine.for %arg8 = 0 to 128 step 128 {
          affine.if affine_set<(d0, d1) : (-d0 - d1 + 127 >= 0)>(%arg7, %arg8) {
            affine.store %cst, %8[%arg7 + %arg8] : memref<128xf32, 2>
            affine.store %cst_0, %9[%arg7 + %arg8] : memref<128xf32, 2>
          }
        }
        affine.for %arg8 = 0 to 2048 step 64 {
          %15 = memref.alloc() : memref<8x8xf32, 3>
          %16 = memref.alloc() : memref<4xf32, 3>
          %17 = memref.alloc() : memref<8xf32, 3>
          affine.for %arg9 = 0 to 8 {
            affine.for %arg10 = 0 to 8 {
              affine.store %cst, %15[%arg9, %arg10] : memref<8x8xf32, 3>
            }
          }
          affine.for %arg9 = 0 to 64 step 8 {
            gpu.barrier
            affine.for %arg10 = 0 to 2 {
              %24 = affine.vector_load %arg0[%arg4, %arg5, %arg6 * 128 + %arg7 floordiv 2 + %arg10 * 64, %arg9 + (%arg7 mod 2) * 4] : memref<8x32x2048x64xf32, 1>, vector<4xf32>
              affine.vector_store %24, %17[%arg10 * 4] : memref<8xf32, 3>, vector<4xf32>
            }
            affine.for %arg10 = 0 to 1 {
              %24 = affine.vector_load %arg1[%arg4, %arg5, %arg8 + %arg7 floordiv 2 + %arg10 * 64, %arg9 + (%arg7 mod 2) * 4] : memref<8x32x2048x64xf32, 1>, vector<4xf32>
              affine.vector_store %24, %16[%arg10 * 4] : memref<4xf32, 3>, vector<4xf32>
            }
            affine.for %arg10 = 0 to 2 {
              affine.for %arg11 = 0 to 4 {
                %24 = affine.vector_load %17[%arg10 * 4 + %arg11] : memref<8xf32, 3>, vector<1xf32>
                affine.vector_store %24, %13[(%arg7 mod 2) * 4 + %arg11, %arg7 floordiv 2 + %arg10 * 64] : memref<8x128xf32, 2>, vector<1xf32>
              }
            }
            affine.for %arg10 = 0 to 1 {
              affine.for %arg11 = 0 to 4 {
                %24 = affine.vector_load %16[%arg10 * 4 + %arg11] : memref<4xf32, 3>, vector<1xf32>
                affine.vector_store %24, %12[(%arg7 mod 2) * 4 + %arg11, %arg7 floordiv 2 + %arg10 * 64] : memref<8x64xf32, 2>, vector<1xf32>
              }
            }
            gpu.barrier
            %22 = memref.alloc() : memref<8xf32, 3>
            %23 = memref.alloc() : memref<8xf32, 3>
            affine.for %arg10 = 0 to 8 {
              affine.for %arg11 = 0 to 2 {
                %24 = affine.vector_load %13[%arg10, (%arg7 floordiv 32) * 32 + %arg11 * 16 + ((%arg7 mod 32) floordiv 8) * 4] : memref<8x128xf32, 2>, vector<4xf32>
                affine.vector_store %24, %22[%arg11 * 4] : memref<8xf32, 3>, vector<4xf32>
              }
              affine.for %arg11 = 0 to 2 {
                %24 = affine.vector_load %12[%arg10, %arg11 * 32 + (%arg7 mod 8) * 4] : memref<8x64xf32, 2>, vector<4xf32>
                affine.vector_store %24, %23[%arg11 * 4] : memref<8xf32, 3>, vector<4xf32>
              }
              affine.for %arg11 = 0 to 8 {
                affine.for %arg12 = 0 to 8 {
                  %24 = affine.load %23[%arg11] : memref<8xf32, 3>
                  %25 = affine.load %22[%arg12] : memref<8xf32, 3>
                  %26 = affine.load %15[%arg11, %arg12] : memref<8x8xf32, 3>
                  %27 = arith.mulf %24, %25 : f32
                  %28 = arith.addf %27, %26 : f32
                  affine.store %28, %15[%arg11, %arg12] : memref<8x8xf32, 3>
                }
              }
            }
          }
          %18 = memref.alloc() : memref<8xf32, 3>
          %19 = memref.alloc() : memref<8xf32, 3>
          affine.for %arg9 = 0 to 8 {
            affine.store %cst_0, %18[%arg9] : memref<8xf32, 3>
            affine.store %cst, %19[%arg9] : memref<8xf32, 3>
            affine.for %arg10 = 0 to 8 {
              %22 = affine.load %18[%arg9] : memref<8xf32, 3>
              %23 = affine.load %15[%arg10, %arg9] : memref<8x8xf32, 3>
              %24 = arith.maxf %22, %23 : f32
              %25 = affine.load %18[%arg9] : memref<8xf32, 3>
              %26 = arith.subf %22, %24 : f32
              %27 = math.exp %26 : f32
              %28 = affine.load %19[%arg9] : memref<8xf32, 3>
              %29 = arith.mulf %27, %28 : f32
              %30 = arith.subf %23, %24 : f32
              %31 = math.exp %30 : f32
              %32 = arith.addf %29, %31 : f32
              affine.store %32, %18[%arg9] : memref<8xf32, 3>
            }
          }
          affine.for %arg9 = 0 to 8 {
            %22 = affine.load %18[%arg9] : memref<8xf32, 3>
            %23 = affine.load %19[%arg9] : memref<8xf32, 3>
            %c8_i32 = arith.constant 8 : i32
            %c1_i32 = arith.constant 1 : i32
            %result, %valid = gpu.shuffle  down %22, %c1_i32, %c8_i32 : f32
            %c8_i32_1 = arith.constant 8 : i32
            %c1_i32_2 = arith.constant 1 : i32
            %result_3, %valid_4 = gpu.shuffle  down %23, %c1_i32_2, %c8_i32_1 : f32
            %24 = arith.maxf %22, %result : f32
            %25 = arith.subf %22, %24 : f32
            %26 = math.exp %25 : f32
            %27 = arith.mulf %23, %26 : f32
            %28 = arith.subf %result, %24 : f32
            %29 = math.exp %28 : f32
            %30 = arith.mulf %result_3, %29 : f32
            %31 = arith.addf %27, %30 : f32
            affine.store %31, %19[%arg9] : memref<8xf32, 3>
            affine.store %24, %18[%arg9] : memref<8xf32, 3>
            %32 = affine.load %18[%arg9] : memref<8xf32, 3>
            %33 = affine.load %19[%arg9] : memref<8xf32, 3>
            %c8_i32_5 = arith.constant 8 : i32
            %c2_i32 = arith.constant 2 : i32
            %result_6, %valid_7 = gpu.shuffle  down %32, %c2_i32, %c8_i32_5 : f32
            %c8_i32_8 = arith.constant 8 : i32
            %c2_i32_9 = arith.constant 2 : i32
            %result_10, %valid_11 = gpu.shuffle  down %33, %c2_i32_9, %c8_i32_8 : f32
            %34 = arith.maxf %32, %result_6 : f32
            %35 = arith.subf %32, %34 : f32
            %36 = math.exp %35 : f32
            %37 = arith.mulf %33, %36 : f32
            %38 = arith.subf %result_6, %34 : f32
            %39 = math.exp %38 : f32
            %40 = arith.mulf %result_10, %39 : f32
            %41 = arith.addf %37, %40 : f32
            affine.store %41, %19[%arg9] : memref<8xf32, 3>
            affine.store %34, %18[%arg9] : memref<8xf32, 3>
            %42 = affine.load %18[%arg9] : memref<8xf32, 3>
            %43 = affine.load %19[%arg9] : memref<8xf32, 3>
            %c8_i32_12 = arith.constant 8 : i32
            %c4_i32 = arith.constant 4 : i32
            %result_13, %valid_14 = gpu.shuffle  down %42, %c4_i32, %c8_i32_12 : f32
            %c8_i32_15 = arith.constant 8 : i32
            %c4_i32_16 = arith.constant 4 : i32
            %result_17, %valid_18 = gpu.shuffle  down %43, %c4_i32_16, %c8_i32_15 : f32
            %44 = arith.maxf %42, %result_13 : f32
            %45 = arith.subf %42, %44 : f32
            %46 = math.exp %45 : f32
            %47 = arith.mulf %43, %46 : f32
            %48 = arith.subf %result_13, %44 : f32
            %49 = math.exp %48 : f32
            %50 = arith.mulf %result_17, %49 : f32
            %51 = arith.addf %47, %50 : f32
            affine.store %51, %19[%arg9] : memref<8xf32, 3>
            affine.store %44, %18[%arg9] : memref<8xf32, 3>
          }
          affine.if affine_set<(d0) : (d0 mod 8 == 0)>(%arg7) {
            affine.for %arg9 = 0 to 8 {
              %22 = affine.load %9[(%arg7 floordiv 32) * 32 + (%arg9 floordiv 4) * 16 + ((%arg7 mod 32) floordiv 8) * 4 + %arg9 mod 4] : memref<128xf32, 2>
              %23 = affine.load %8[(%arg7 floordiv 32) * 32 + (%arg9 floordiv 4) * 16 + ((%arg7 mod 32) floordiv 8) * 4 + %arg9 mod 4] : memref<128xf32, 2>
              %24 = affine.load %18[%arg9] : memref<8xf32, 3>
              %25 = arith.maxf %22, %24 : f32
              %26 = arith.subf %24, %25 : f32
              %27 = math.exp %26 : f32
              affine.store %25, %9[(%arg7 floordiv 32) * 32 + (%arg9 floordiv 4) * 16 + ((%arg7 mod 32) floordiv 8) * 4 + %arg9 mod 4] : memref<128xf32, 2>
              affine.store %27, %7[(%arg7 floordiv 32) * 32 + (%arg9 floordiv 4) * 16 + ((%arg7 mod 32) floordiv 8) * 4 + %arg9 mod 4] : memref<128xf32, 2>
              %28 = arith.mulf %23, %27 : f32
              %29 = affine.load %19[%arg9] : memref<8xf32, 3>
              %30 = arith.subf %24, %25 : f32
              %31 = math.exp %30 : f32
              %32 = arith.mulf %29, %31 : f32
              %33 = arith.addf %28, %32 : f32
              affine.store %33, %8[(%arg7 floordiv 32) * 32 + (%arg9 floordiv 4) * 16 + ((%arg7 mod 32) floordiv 8) * 4 + %arg9 mod 4] : memref<128xf32, 2>
              affine.store %25, %18[%arg9] : memref<8xf32, 3>
            }
          }
          affine.for %arg9 = 0 to 8 {
            %22 = affine.load %18[%arg9] : memref<8xf32, 3>
            %c8_i32 = arith.constant 8 : i32
            %c0_i32 = arith.constant 0 : i32
            %result, %valid = gpu.shuffle  idx %22, %c0_i32, %c8_i32 : f32
            affine.store %result, %18[%arg9] : memref<8xf32, 3>
          }
          affine.for %arg9 = 0 to 8 {
            affine.for %arg10 = 0 to 8 {
              %22 = affine.load %15[%arg9, %arg10] : memref<8x8xf32, 3>
              %23 = affine.load %18[%arg10] : memref<8xf32, 3>
              %24 = arith.subf %22, %23 : f32
              %25 = math.exp %24 : f32
              affine.store %25, %15[%arg9, %arg10] : memref<8x8xf32, 3>
            }
          }
          affine.for %arg9 = 0 to 8 {
            affine.for %arg10 = 0 to 8 step 4 {
              %22 = affine.vector_load %15[%arg9, %arg10] : memref<8x8xf32, 3>, vector<4xf32>
              affine.vector_store %22, %10[(%arg9 floordiv 4) * 32 + (%arg7 mod 8) * 4 + %arg9 mod 4, (%arg7 floordiv 32) * 32 + %arg10 * 4 + ((%arg7 mod 32) floordiv 8) * 4] : memref<128x64xf32, 2>, vector<4xf32>
            }
          }
          gpu.barrier
          %20 = memref.alloc() : memref<8xf32, 3>
          affine.for %arg9 = 0 to 2 {
            %22 = affine.vector_load %7[((%arg7 floordiv 32) floordiv 2) * 64 + %arg9 * 32 + ((%arg7 mod 32) floordiv 4) * 4] : memref<128xf32, 2>, vector<4xf32>
            affine.vector_store %22, %20[%arg9 * 4] : memref<8xf32, 3>, vector<4xf32>
          }
          affine.for %arg9 = 0 to 8 {
            affine.for %arg10 = 0 to 8 {
              %22 = affine.load %6[%arg9, %arg10] : memref<8x8xf32, 3>
              %23 = affine.load %20[%arg9] : memref<8xf32, 3>
              %24 = arith.mulf %22, %23 : f32
              affine.store %24, %6[%arg9, %arg10] : memref<8x8xf32, 3>
            }
          }
          %21 = memref.alloc() : memref<4xf32, 3>
          affine.for %arg9 = 0 to 64 step 8 {
            gpu.barrier
            affine.for %arg10 = 0 to 1 {
              %24 = affine.vector_load %arg2[%arg4, %arg5, %arg8 + %arg7 floordiv 16 + %arg10 * 8 + %arg9, (%arg7 mod 16) * 4] : memref<8x32x2048x64xf32, 1>, vector<4xf32>
              affine.vector_store %24, %21[%arg10 * 4] : memref<4xf32, 3>, vector<4xf32>
            }
            affine.for %arg10 = 0 to 1 {
              %24 = affine.vector_load %21[%arg10 * 4] : memref<4xf32, 3>, vector<4xf32>
              affine.vector_store %24, %11[%arg7 floordiv 16 + %arg10 * 8, (%arg7 mod 16) * 4] : memref<8x64xf32, 2>, vector<4xf32>
            }
            gpu.barrier
            %22 = memref.alloc() : memref<8xf32, 3>
            %23 = memref.alloc() : memref<8xf32, 3>
            affine.for %arg10 = 0 to 8 {
              affine.for %arg11 = 0 to 2 {
                %24 = affine.vector_load %10[%arg9 + %arg10, ((%arg7 floordiv 32) floordiv 2) * 64 + %arg11 * 32 + ((%arg7 mod 32) floordiv 4) * 4] : memref<128x64xf32, 2>, vector<4xf32>
                affine.vector_store %24, %22[%arg11 * 4] : memref<8xf32, 3>, vector<4xf32>
              }
              affine.for %arg11 = 0 to 2 {
                %24 = affine.vector_load %11[%arg10, ((%arg7 floordiv 32) mod 2) * 32 + %arg11 * 16 + (%arg7 mod 4) * 4] : memref<8x64xf32, 2>, vector<4xf32>
                affine.vector_store %24, %23[%arg11 * 4] : memref<8xf32, 3>, vector<4xf32>
              }
              affine.for %arg11 = 0 to 8 {
                affine.for %arg12 = 0 to 8 {
                  %24 = affine.load %22[%arg11] : memref<8xf32, 3>
                  %25 = affine.load %23[%arg12] : memref<8xf32, 3>
                  %26 = affine.load %6[%arg11, %arg12] : memref<8x8xf32, 3>
                  %27 = arith.mulf %24, %25 : f32
                  %28 = arith.addf %27, %26 : f32
                  affine.store %28, %6[%arg11, %arg12] : memref<8x8xf32, 3>
                }
              }
            }
          }
        }
        %14 = memref.alloc() : memref<8xf32, 3>
        affine.for %arg8 = 0 to 2 {
          %15 = affine.vector_load %8[((%arg7 floordiv 32) floordiv 2) * 64 + %arg8 * 32 + ((%arg7 mod 32) floordiv 4) * 4] : memref<128xf32, 2>, vector<4xf32>
          affine.vector_store %15, %14[%arg8 * 4] : memref<8xf32, 3>, vector<4xf32>
        }
        affine.for %arg8 = 0 to 8 {
          affine.for %arg9 = 0 to 8 {
            %15 = affine.load %6[%arg8, %arg9] : memref<8x8xf32, 3>
            %16 = affine.load %14[%arg8] : memref<8xf32, 3>
            %17 = arith.divf %15, %16 : f32
            affine.store %17, %6[%arg8, %arg9] : memref<8x8xf32, 3>
          }
        }
        affine.for %arg8 = 0 to 8 {
          affine.for %arg9 = 0 to 8 step 4 {
            %15 = affine.vector_load %6[%arg8, %arg9] : memref<8x8xf32, 3>, vector<4xf32>
            affine.vector_store %15, %arg3[%arg4, %arg5, %arg6 * 128 + ((%arg4 floordiv 32) floordiv 2) * 64 + (%arg8 floordiv 4) * 32 + ((%arg4 mod 32) floordiv 4) * 4 + %arg8 mod 4, ((%arg4 floordiv 32) mod 2) * 32 + %arg9 * 4 + (%arg4 mod 4) * 4] : memref<8x32x2048x64xf32, 1>, vector<4xf32>
          }
        }
      }
    }
    return %arg3 : memref<8x32x2048x64xf32, 1>
  }
  func.func @BatchMatmul_8_32_m2048_n64_k2048_NN(%arg0: memref<8x32x2048x2048xf32, 1>, %arg1: memref<8x32x2048x64xf32, 1>, %arg2: memref<8x32x2048x64xf32, 1>) -> memref<8x32x2048x64xf32, 1> attributes {func.state = "cpu"} {
    affine.for %arg3 = 0 to 8 {
      affine.for %arg4 = 0 to 32 {
        affine.for %arg5 = 0 to 2048 {
          affine.for %arg6 = 0 to 64 {
            %cst = arith.constant 0.000000e+00 : f32
            %6 = affine.for %arg7 = 0 to 2048 iter_args(%arg8 = %cst) -> (f32) {
              %7 = affine.load %arg0[%arg3, %arg4, %arg5, %arg7] : memref<8x32x2048x2048xf32, 1>
              %8 = affine.load %arg1[%arg3, %arg4, %arg7, %arg6] : memref<8x32x2048x64xf32, 1>
              %9 = arith.mulf %7, %8 : f32
              %10 = arith.addf %9, %arg8 : f32
              affine.yield %10 : f32
            }
            affine.store %6, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<8x32x2048x64xf32, 1>
          }
        }
      }
    }
    return %arg2 : memref<8x32x2048x64xf32, 1>
  }
  func.func @Softmax_8_32_2048_2048_axis3(%arg0: memref<8x32x2048x2048xf32, 1>) -> memref<8x32x2048x2048xf32, 1> attributes {func.state = "cpu"} {
    affine.for %arg1 = 0 to 8 {
      affine.for %arg2 = 0 to 32 {
        affine.for %arg3 = 0 to 2048 {
          %cst = arith.constant 0.000000e+00 : f32
          %6 = affine.for %arg4 = 0 to 2048 iter_args(%arg5 = %cst) -> (f32) {
            %7 = affine.load %arg0[%arg1, %arg2, %arg3, %arg4] : memref<8x32x2048x2048xf32, 1>
            %8 = math.exp %7 : f32
            %9 = arith.addf %8, %arg5 : f32
            affine.yield %9 : f32
          }
          affine.for %arg4 = 0 to 2048 {
            %7 = affine.load %arg0[%arg1, %arg2, %arg3, %arg4] : memref<8x32x2048x2048xf32, 1>
            %8 = math.exp %7 : f32
            %9 = arith.divf %8, %6 : f32
            affine.store %9, %arg0[%arg1, %arg2, %arg3, %arg4] : memref<8x32x2048x2048xf32, 1>
          }
        }
      }
    }
    return %arg0 : memref<8x32x2048x2048xf32, 1>
  }
  func.func @BatchMatmul_8_32_m2048_n2048_k64_NT(%arg0: memref<8x32x2048x64xf32, 1>, %arg1: memref<8x32x2048x64xf32, 1>, %arg2: memref<8x32x2048x2048xf32, 1>) -> memref<8x32x2048x2048xf32, 1> attributes {func.state = "cpu"} {
    affine.for %arg3 = 0 to 8 {
      affine.for %arg4 = 0 to 32 {
        affine.for %arg5 = 0 to 2048 {
          affine.for %arg6 = 0 to 2048 {
            %cst = arith.constant 0.000000e+00 : f32
            %6 = affine.for %arg7 = 0 to 64 iter_args(%arg8 = %cst) -> (f32) {
              %7 = affine.load %arg0[%arg3, %arg4, %arg5, %arg7] : memref<8x32x2048x64xf32, 1>
              %8 = affine.load %arg1[%arg3, %arg4, %arg6, %arg7] : memref<8x32x2048x64xf32, 1>
              %9 = arith.mulf %7, %8 : f32
              %10 = arith.addf %9, %arg8 : f32
              affine.yield %10 : f32
            }
            affine.store %6, %arg2[%arg3, %arg4, %arg5, %arg6] : memref<8x32x2048x2048xf32, 1>
          }
        }
      }
    }
    return %arg2 : memref<8x32x2048x2048xf32, 1>
  }



// grid dims:(8, 32, 16, ), block dims:(128, )
__global__ void kernel0(float* arg0, float* arg1, float* arg2, float* arg3) {
  float array0[8][8];
  __shared__ float array1[128];
  __shared__ float array2[128];
  __shared__ float array3[128];
  __shared__ float array4[128][64];
  __shared__ float array5[8][64];
  __shared__ float array6[8][64];
  __shared__ float array7[8][128];
  constexpr float const0th = 0;
  constexpr float const1th = -3.40282e+38;
  for (int iter0 = 0; iter0 < 8; iter0 += 1) {
    for (int iter1 = 0; iter1 < 8; iter1 += 1) {
      array0[iter0][iter1] = const0th;
    }
  }
  for (int iter2 = 0; iter2 < 128; iter2 += 128) {
    if ((((threadIdx.x * -1) + (iter2 * -1)) + 127) >= 0 &&  true) {
      array2[(threadIdx.x + iter2)] = const0th;
      array3[(threadIdx.x + iter2)] = const1th;
    }
  }
  for (int iter3 = 0; iter3 < 2048; iter3 += 64) {
    float array8[8][8];
    float array9[4];
    float array10[8];
    for (int iter4 = 0; iter4 < 8; iter4 += 1) {
      for (int iter5 = 0; iter5 < 8; iter5 += 1) {
        array8[iter4][iter5] = const0th;
      }
    }
    for (int iter6 = 0; iter6 < 64; iter6 += 8) {
      __syncthreads();
      for (int iter7 = 0; iter7 < 2; iter7 += 1) {
        auto vec0 = (reinterpret_cast<float4*>(&(arg0[blockIdx.z * 4194304 + blockIdx.y * 131072 + ((blockIdx.x * 128) + ((threadIdx.x / 2) + (iter7 * 64))) * 64 + (iter6 + ((threadIdx.x % 2) * 4)) * 1 + 0]))[0]);
        (reinterpret_cast<float4*>(&(array10[(iter7 * 4)]))[0]) = vec0;
      }
      for (int iter8 = 0; iter8 < 1; iter8 += 1) {
        auto vec1 = (reinterpret_cast<float4*>(&(arg1[blockIdx.z * 4194304 + blockIdx.y * 131072 + (iter3 + ((threadIdx.x / 2) + (iter8 * 64))) * 64 + (iter6 + ((threadIdx.x % 2) * 4)) * 1 + 0]))[0]);
        (reinterpret_cast<float4*>(&(array9[(iter8 * 4)]))[0]) = vec1;
      }
      for (int iter9 = 0; iter9 < 2; iter9 += 1) {
        for (int iter10 = 0; iter10 < 4; iter10 += 1) {
          auto vec2 = (reinterpret_cast<float1*>(&(array10[((iter9 * 4) + iter10)]))[0]);
          (reinterpret_cast<float1*>(&(array7[(((threadIdx.x % 2) * 4) + iter10)][((threadIdx.x / 2) + (iter9 * 64))]))[0]) = vec2;
        }
      }
      for (int iter11 = 0; iter11 < 1; iter11 += 1) {
        for (int iter12 = 0; iter12 < 4; iter12 += 1) {
          auto vec3 = (reinterpret_cast<float1*>(&(array9[((iter11 * 4) + iter12)]))[0]);
          (reinterpret_cast<float1*>(&(array6[(((threadIdx.x % 2) * 4) + iter12)][((threadIdx.x / 2) + (iter11 * 64))]))[0]) = vec3;
        }
      }
      __syncthreads();
      float array11[8];
      float array12[8];
      for (int iter13 = 0; iter13 < 8; iter13 += 1) {
        for (int iter14 = 0; iter14 < 2; iter14 += 1) {
          auto vec4 = (reinterpret_cast<float4*>(&(array7[iter13][((((threadIdx.x / 32) * 32) + (iter14 * 16)) + (((threadIdx.x % 32) / 8) * 4))]))[0]);
          (reinterpret_cast<float4*>(&(array11[(iter14 * 4)]))[0]) = vec4;
        }
        for (int iter15 = 0; iter15 < 2; iter15 += 1) {
          auto vec5 = (reinterpret_cast<float4*>(&(array6[iter13][((iter15 * 32) + ((threadIdx.x % 8) * 4))]))[0]);
          (reinterpret_cast<float4*>(&(array12[(iter15 * 4)]))[0]) = vec5;
        }
        for (int iter16 = 0; iter16 < 8; iter16 += 1) {
          for (int iter17 = 0; iter17 < 8; iter17 += 1) {
            auto R0 = array12[iter16];
            auto R1 = array11[iter17];
            auto R2 = array8[iter16][iter17];
            auto temp0 = R0 * R1;
            auto temp12 = temp0 + R2;
            array8[iter16][iter17] = temp12;
          }
        }
      }
    }
    float array13[8];
    float array14[8];
    for (int iter18 = 0; iter18 < 8; iter18 += 1) {
      array13[iter18] = const1th;
      array14[iter18] = const0th;
      for (int iter19 = 0; iter19 < 8; iter19 += 1) {
        auto R3 = array13[iter18];
        auto R4 = array8[iter19][iter18];
        auto temp19 = max(R3 , R4);
        auto R5 = array13[iter18];
        auto temp24 = R3 - temp19;
        auto temp36 = exp(temp24);
        auto R6 = array14[iter18];
        auto temp1 = temp36 * R6;
        auto temp25 = R4 - temp19;
        auto temp37 = exp(temp25);
        auto temp13 = temp1 + temp37;
        array13[iter18] = temp13;
      }
    }
    for (int iter20 = 0; iter20 < 8; iter20 += 1) {
      auto R7 = array13[iter20];
      auto R8 = array14[iter20];
      constexpr int const2th = 8;
      constexpr int const3th = 1;
      auto temp47 =  __shfl_down_sync(0xffffffff, R7, const3th, const2th);
      constexpr int const4th = 8;
      constexpr int const5th = 1;
      auto temp48 =  __shfl_down_sync(0xffffffff, R8, const5th, const4th);
      auto temp20 = max(R7 , temp47);
      auto temp26 = R7 - temp20;
      auto temp38 = exp(temp26);
      auto temp2 = R8 * temp38;
      auto temp27 = temp47 - temp20;
      auto temp39 = exp(temp27);
      auto temp3 = temp48 * temp39;
      auto temp14 = temp2 + temp3;
      array14[iter20] = temp14;
      array13[iter20] = temp20;
      auto R9 = array13[iter20];
      auto R10 = array14[iter20];
      constexpr int const6th = 8;
      constexpr int const7th = 2;
      auto temp49 =  __shfl_down_sync(0xffffffff, R9, const7th, const6th);
      constexpr int const8th = 8;
      constexpr int const9th = 2;
      auto temp50 =  __shfl_down_sync(0xffffffff, R10, const9th, const8th);
      auto temp21 = max(R9 , temp49);
      auto temp28 = R9 - temp21;
      auto temp40 = exp(temp28);
      auto temp4 = R10 * temp40;
      auto temp29 = temp49 - temp21;
      auto temp41 = exp(temp29);
      auto temp5 = temp50 * temp41;
      auto temp15 = temp4 + temp5;
      array14[iter20] = temp15;
      array13[iter20] = temp21;
      auto R11 = array13[iter20];
      auto R12 = array14[iter20];
      constexpr int const10th = 8;
      constexpr int const11th = 4;
      auto temp51 =  __shfl_down_sync(0xffffffff, R11, const11th, const10th);
      constexpr int const12th = 8;
      constexpr int const13th = 4;
      auto temp52 =  __shfl_down_sync(0xffffffff, R12, const13th, const12th);
      auto temp22 = max(R11 , temp51);
      auto temp30 = R11 - temp22;
      auto temp42 = exp(temp30);
      auto temp6 = R12 * temp42;
      auto temp31 = temp51 - temp22;
      auto temp43 = exp(temp31);
      auto temp7 = temp52 * temp43;
      auto temp16 = temp6 + temp7;
      array14[iter20] = temp16;
      array13[iter20] = temp22;
    }
    if ((threadIdx.x % 8) == 0 &&  true) {
      for (int iter21 = 0; iter21 < 8; iter21 += 1) {
        auto R13 = array3[(((((threadIdx.x / 32) * 32) + ((iter21 / 4) * 16)) + (((threadIdx.x % 32) / 8) * 4)) + (iter21 % 4))];
        auto R14 = array2[(((((threadIdx.x / 32) * 32) + ((iter21 / 4) * 16)) + (((threadIdx.x % 32) / 8) * 4)) + (iter21 % 4))];
        auto R15 = array13[iter21];
        auto temp23 = max(R13 , R15);
        auto temp32 = R15 - temp23;
        auto temp44 = exp(temp32);
        array3[(((((threadIdx.x / 32) * 32) + ((iter21 / 4) * 16)) + (((threadIdx.x % 32) / 8) * 4)) + (iter21 % 4))] = temp23;
        array1[(((((threadIdx.x / 32) * 32) + ((iter21 / 4) * 16)) + (((threadIdx.x % 32) / 8) * 4)) + (iter21 % 4))] = temp44;
        auto temp8 = R14 * temp44;
        auto R16 = array14[iter21];
        auto temp33 = R15 - temp23;
        auto temp45 = exp(temp33);
        auto temp9 = R16 * temp45;
        auto temp17 = temp8 + temp9;
        array2[(((((threadIdx.x / 32) * 32) + ((iter21 / 4) * 16)) + (((threadIdx.x % 32) / 8) * 4)) + (iter21 % 4))] = temp17;
        array13[iter21] = temp23;
      }
    }
    for (int iter22 = 0; iter22 < 8; iter22 += 1) {
      auto R17 = array13[iter22];
      constexpr int const14th = 8;
      constexpr int const15th = 0;
      auto temp53 =  __shfl_sync(0xffffffff, R17, const15th, const14th);
      array13[iter22] = temp53;
    }
    for (int iter23 = 0; iter23 < 8; iter23 += 1) {
      for (int iter24 = 0; iter24 < 8; iter24 += 1) {
        auto R18 = array8[iter23][iter24];
        auto R19 = array13[iter24];
        auto temp34 = R18 - R19;
        auto temp46 = exp(temp34);
        array8[iter23][iter24] = temp46;
      }
    }
    for (int iter25 = 0; iter25 < 8; iter25 += 1) {
      for (int iter26 = 0; iter26 < 8; iter26 += 4) {
        auto vec6 = (reinterpret_cast<float4*>(&(array8[iter25][iter26]))[0]);
        (reinterpret_cast<float4*>(&(array4[((((iter25 / 4) * 32) + ((threadIdx.x % 8) * 4)) + (iter25 % 4))][((((threadIdx.x / 32) * 32) + (iter26 * 4)) + (((threadIdx.x % 32) / 8) * 4))]))[0]) = vec6;
      }
    }
    __syncthreads();
    float array15[8];
    for (int iter27 = 0; iter27 < 2; iter27 += 1) {
      auto vec7 = (reinterpret_cast<float4*>(&(array1[(((((threadIdx.x / 32) / 2) * 64) + (iter27 * 32)) + (((threadIdx.x % 32) / 4) * 4))]))[0]);
      (reinterpret_cast<float4*>(&(array15[(iter27 * 4)]))[0]) = vec7;
    }
    for (int iter28 = 0; iter28 < 8; iter28 += 1) {
      for (int iter29 = 0; iter29 < 8; iter29 += 1) {
        auto R20 = array0[iter28][iter29];
        auto R21 = array15[iter28];
        auto temp10 = R20 * R21;
        array0[iter28][iter29] = temp10;
      }
    }
    float array16[4];
    for (int iter30 = 0; iter30 < 64; iter30 += 8) {
      __syncthreads();
      for (int iter31 = 0; iter31 < 1; iter31 += 1) {
        auto vec8 = (reinterpret_cast<float4*>(&(arg2[blockIdx.z * 4194304 + blockIdx.y * 131072 + ((iter3 + ((threadIdx.x / 16) + (iter31 * 8))) + iter30) * 64 + ((threadIdx.x % 16) * 4) * 1 + 0]))[0]);
        (reinterpret_cast<float4*>(&(array16[(iter31 * 4)]))[0]) = vec8;
      }
      for (int iter32 = 0; iter32 < 1; iter32 += 1) {
        auto vec9 = (reinterpret_cast<float4*>(&(array16[(iter32 * 4)]))[0]);
        (reinterpret_cast<float4*>(&(array5[((threadIdx.x / 16) + (iter32 * 8))][((threadIdx.x % 16) * 4)]))[0]) = vec9;
      }
      __syncthreads();
      float array17[8];
      float array18[8];
      for (int iter33 = 0; iter33 < 8; iter33 += 1) {
        for (int iter34 = 0; iter34 < 2; iter34 += 1) {
          auto vec10 = (reinterpret_cast<float4*>(&(array4[(iter30 + iter33)][(((((threadIdx.x / 32) / 2) * 64) + (iter34 * 32)) + (((threadIdx.x % 32) / 4) * 4))]))[0]);
          (reinterpret_cast<float4*>(&(array17[(iter34 * 4)]))[0]) = vec10;
        }
        for (int iter35 = 0; iter35 < 2; iter35 += 1) {
          auto vec11 = (reinterpret_cast<float4*>(&(array5[iter33][(((((threadIdx.x / 32) % 2) * 32) + (iter35 * 16)) + ((threadIdx.x % 4) * 4))]))[0]);
          (reinterpret_cast<float4*>(&(array18[(iter35 * 4)]))[0]) = vec11;
        }
        for (int iter36 = 0; iter36 < 8; iter36 += 1) {
          for (int iter37 = 0; iter37 < 8; iter37 += 1) {
            auto R22 = array17[iter36];
            auto R23 = array18[iter37];
            auto R24 = array0[iter36][iter37];
            auto temp11 = R22 * R23;
            auto temp18 = temp11 + R24;
            array0[iter36][iter37] = temp18;
          }
        }
      }
    }
  }
  float array19[8];
  for (int iter38 = 0; iter38 < 2; iter38 += 1) {
    auto vec12 = (reinterpret_cast<float4*>(&(array2[(((((threadIdx.x / 32) / 2) * 64) + (iter38 * 32)) + (((threadIdx.x % 32) / 4) * 4))]))[0]);
    (reinterpret_cast<float4*>(&(array19[(iter38 * 4)]))[0]) = vec12;
  }
  for (int iter39 = 0; iter39 < 8; iter39 += 1) {
    for (int iter40 = 0; iter40 < 8; iter40 += 1) {
      auto R25 = array0[iter39][iter40];
      auto R26 = array19[iter39];
      auto temp35 = R25 / R26;
      array0[iter39][iter40] = temp35;
    }
  }
  for (int iter41 = 0; iter41 < 8; iter41 += 1) {
    for (int iter42 = 0; iter42 < 8; iter42 += 4) {
      auto vec13 = (reinterpret_cast<float4*>(&(array0[iter41][iter42]))[0]);
      (reinterpret_cast<float4*>(&(arg3[blockIdx.z * 4194304 + blockIdx.y * 131072 + ((blockIdx.x * 128) + ((((((blockIdx.z / 32) / 2) * 64) + ((iter41 / 4) * 32)) + (((blockIdx.z % 32) / 4) * 4)) + (iter41 % 4))) * 64 + (((((blockIdx.z / 32) % 2) * 32) + (iter42 * 4)) + ((blockIdx.z % 4) * 4)) * 1 + 0]))[0]) = vec13;
    }
  }
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