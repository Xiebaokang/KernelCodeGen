#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"
using namespace KernelCodeGen;


void test_binary() {
  KernelCodeGenerator generator("CUDA");
  auto graph = generator.createGraph("demo");
  generator.setLogMode(Log::Debug);

  // auto indices = graph.create<PlaceHolder>(std::vector<int64_t>{2, 20}, std::string{"index"});
  // auto data = graph.create<PlaceHolder>(std::vector<int64_t>{15, 256}, std::string{"float32"});
  // auto gather1 = graph.create<Gather>(data, indices);
  // auto gather2 = graph.create<Gather>(data, 0, 1);  // indices:0  axis:1


  // auto A = graph.create<PlaceHolder>(std::vector<int64_t>{1, 2048, 64}, std::string{"float32"});
  // auto B = graph.create<PlaceHolder>(std::vector<int64_t>{1, 2048, 64}, std::string{"float32"});
  // auto add = graph.create<Binary>(A, B, "add", MemorySpace::global);

  // std::vector<int> dim1{1, 8, 16};
  // std::vector<int> dim2{1024, 2048};
  // std::vector<std::vector<mlir::Value>> tensors;
  // for (auto d1 : dim1) {
  //   for (auto d2 : dim2) {
  //     auto A = graph.create<PlaceHolder>(std::vector<int64_t>{d1, d2, 64}, std::string{"float32"});
  //     auto B = graph.create<PlaceHolder>(std::vector<int64_t>{d1, d2, 64}, std::string{"float32"});
  //     tensors.push_back({A, B});
  //   }
  // }
  // for (auto tensor : tensors) {
  //   auto add = graph.create<Binary>(tensor[0], tensor[1], "add", MemorySpace::global);
  // }

  auto A = graph.create<PlaceHolder>(std::vector<int64_t>{16, 2048, 64}, std::string{"float32"});
  // auto tanh = graph.create<ElementWise>(A, "tanh", MemorySpace::inplace);
  // auto relu = graph.create<ElementWise>(A, "gelu", MemorySpace::inplace);
  // auto cast = graph.create<ElementWise>(A, "cast", MemorySpace::global, "int32");
  std::vector<int64_t> axes = {1, 2};
  auto cast = graph.create<LayerNorm>(A, axes, MemorySpace::inplace);

  graph.dump();

  auto module = generator.optimize(graph);
  generator.dump(module);
  auto&& sourceCode = generator.codegen(module);
  // generator.save(sourceCode, "../test/matmul/matmulKernel.cu");
}

void test_matmul() {

  /* 1. Demo */
  KernelCodeGenerator generator("CUDA");
  
  auto graph = generator.createGraph("matmul_demo");
  generator.setLogMode(Log::Debug);

  int m = 4096, n = 2048, k = 1024;
  auto A = graph.create<PlaceHolder>(std::vector<int64_t>{m, k}, std::string{"float32"});
  auto B = graph.create<PlaceHolder>(std::vector<int64_t>{k, n}, std::string{"float32"});
  auto C = graph.create<Matmul>(A, B, MemorySpace::global);
  auto D = graph.create<Relu>(C, MemorySpace::inplace);

  graph.dump();
  auto module = generator.optimize(graph);
  generator.dump(module);
  auto&& sourceCode = generator.codegen(module);
  generator.save(sourceCode, "../test/matmul/matmulKernel.cu");
  std::string adaptorCode = "";
  adaptorCode += "#include \"matmulKernel.cu\"\n";
  adaptorCode += "const int M = " + std::to_string(m) + ";\n";
  adaptorCode += "const int N = " + std::to_string(n) + ";\n";
  adaptorCode += "const int K = " + std::to_string(k) + ";\n";
  adaptorCode += "#define kernelFunc matmul_demo::kernel0\n";
  generator.save(adaptorCode, "../test/matmul/adaptor.cu");
  system("cd ../build && make matmul && ../bin/matmul");

  /* 2. Batch perf test.*/
  std::vector<int64_t> dims {256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384};
  for (auto dim : dims) {
    auto fileName = "Matmul_M" + std::to_string(dim) + "_N" + std::to_string(dim) + "_K" + std::to_string(dim);
    auto graph = generator.createGraph(fileName);
    generator.setLogMode(Log::Release);

    int m = dim, n = dim, k = dim;
    auto A = graph.create<PlaceHolder>(std::vector<int64_t>{m, k}, std::string{"float32"});
    auto B = graph.create<PlaceHolder>(std::vector<int64_t>{k, n}, std::string{"float32"});
    auto C = graph.create<Matmul>(A, B, MemorySpace::global);

    auto module = generator.optimize(graph);
    auto&& sourceCode = generator.codegen(module);
    auto srcFile = "../test/matmul/" + fileName + ".cu";
    generator.save(sourceCode, srcFile);

    std::string adaptorCode = "";
    adaptorCode += "#include \"" + fileName + ".cu\"\n";
    adaptorCode += "const int M = " + std::to_string(m) + ";\n";
    adaptorCode += "const int N = " + std::to_string(n) + ";\n";
    adaptorCode += "const int K = " + std::to_string(k) + ";\n";
    adaptorCode += "#define kernelFunc " + fileName + "::kernel0\n";
    generator.save(adaptorCode, "../test/matmul/adaptor.cu");
    system("cd ../build && make matmul && ../bin/matmul");
  }
}

void test_flash_attention() {
  /* 1. Demo */
  KernelCodeGenerator generator("CUDA");
  
  auto graph = generator.createGraph("flash_attn_demo");
  generator.setLogMode(Log::Debug);

  int64_t hidden_dim = 2048L;
  int64_t total_token = 16 * 1024L;
  int64_t head_dim = 64L;
  int64_t seq_len = 2048L;

  int64_t batch_size = total_token / seq_len;
  int64_t head_num = hidden_dim / head_dim;
  

  auto Q = graph.create<PlaceHolder>(std::vector<int64_t>{
    batch_size, head_num, seq_len, head_dim}, std::string{"float32"});
  auto K = graph.create<PlaceHolder>(std::vector<int64_t>{
    batch_size, head_num, seq_len, head_dim}, std::string{"float32"});
  
  auto V = graph.create<PlaceHolder>(std::vector<int64_t>{
    batch_size, head_num, seq_len, head_dim}, std::string{"float32"});

  auto S = graph.create<BatchedMatmul>(Q, Layout::rowMajor, K, Layout::colMajor);
  auto P = graph.create<Softmax>(S, -1, MemorySpace::inplace);
  auto O = graph.create<BatchedMatmul>(P, Layout::rowMajor, V, Layout::rowMajor);
  

  graph.dump();
  auto module = generator.optimize(graph);
  generator.dump(module);
  auto&& sourceCode = generator.codegen(module);
  generator.save(sourceCode, "../test/flash-attn/demo.cu");
  // std::string adaptorCode = "";
  // adaptorCode += "#include \"matmulKernel.cu\"\n";
  // adaptorCode += "const int M = " + std::to_string(m) + ";\n";
  // adaptorCode += "const int N = " + std::to_string(n) + ";\n";
  // adaptorCode += "const int K = " + std::to_string(k) + ";\n";
  // adaptorCode += "#define kernelFunc matmul_demo::kernel0\n";
  // generator.save(adaptorCode, "../test/matmul/adaptor.cu");
  // system("cd ../build && make matmul && ../bin/matmul");

}


int main(int argc, char* argv[]) {

  // test_matmul();
  test_binary();

  test_flash_attention();

}