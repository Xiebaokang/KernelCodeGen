#pragma once

#include "IR/IR.h"
#include "enum.h"
#include "log.h"

namespace KernelCodeGen {

using loopfunc = std::function<mlir::Value(mlir::OpBuilder&, mlir::Location, mlir::ValueRange, mlir::ValueRange)>;

mlir::func::FuncOp buildFuction(mlir::ModuleOp module, mlir::OpBuilder& builder, 
 const std::string& funcName, const std::vector<mlir::Type>& inputsTypes, 
 const std::vector<mlir::Type>& outputsTypes);
 
// Responsible for the construction of the graph,
//  and store graph module. 
struct ComputeDAG {
  ComputeDAG(mlir::OpBuilder& builder_) : builder(builder_) {};
  ComputeDAG() = default;
  template <typename OperatorType, typename... Args>
  mlir::Value create(Args &&...args) {
    // auto block = builder.getInsertionBlock();
    // auto iter = builder.getInsertionPoint();
    mlir::Value result;
    {
      // mlir::OpBuilder::InsertionGuard guard(builder);
      //Need to gurantee that OperatorType::build only create a nested AffineForOp or AllocOp.
      result = OperatorType::build(this, std::forward<Args>(args)...);
    }
    // builder.setInsertionPoint(block, ++(++iter));
    return result;
  }

  void dump(const std::string& info = "") {
    if (KCGLog::level == Log::Release) return;
    llvm::errs() << "-----------------------------------------------------------\n";
    llvm::errs() << "           " << info << "\n";
    llvm::errs() << "-----------------------------------------------------------\n";
    module->dump();
    if (mlir::failed(mlir::verify(module))) {
      module->emitError("graph verification error");
      assert(false);
    }
  }

  // ComputeDAG& operator=(const ComputeDAG& other) {
  //   if (module != other.module) {
  //     module = other.module;
  //   }
  //   return *this;
  // } 

  // reference to KernelCodeGenerator::builder.
  mlir::OpBuilder builder;
  mlir::ModuleOp module;
};

mlir::Type getDType(mlir::OpBuilder& builder, const std::string& dtype);

/*
// Call like this
KernelCodeGenerator.ComputeDAG.create<>(args...);
*/

// Interface to define operators.
template <typename T>
struct Operator {
  template <typename... Args>
  static mlir::Value build(ComputeDAG* graph, Args &&...args) {
    return T::build(graph, std::forward<Args>(args)...);
  }
};


struct PlaceHolder : Operator<PlaceHolder> {
  static mlir::Value build(ComputeDAG* graph, const std::vector<int64_t>& shapes, const std::string& dtype);
};

struct Matmul : Operator<Matmul> {
  static mlir::Value build(ComputeDAG* graph, mlir::Value A, mlir::Value B/*, MemorySpace ms*/, const std::string& dtype = {""});
};

struct Relu : Operator<Relu> {
  static mlir::Value build(ComputeDAG* graph, mlir::Value input, MemorySpace ms, const std::string& dtype = {""});
};

struct BatchedMatmul : Operator<BatchedMatmul> {
  static mlir::Value build(ComputeDAG* graph, mlir::Value A, Layout layoutA, mlir::Value B, Layout layoutB, const std::string& dtype = {""});
};

// Compute the sum of the lowest dims and divide the result elementwisely.
// If the axis = 1, the reduction scope is the last dim.
struct Softmax : Operator<Softmax> {
  static mlir::Value build(ComputeDAG* graph, mlir::Value input, int axis = -1, MemorySpace ms = MemorySpace::global, const std::string& dtype = {""});
};

struct Transpose : Operator<Transpose> {
  static mlir::Value build(ComputeDAG* graph, mlir::Value input);
};

struct Binary : Operator<Binary> {
  static mlir::Value build(ComputeDAG* graph, mlir::Value A, mlir::Value B, std::string operation, MemorySpace ms=MemorySpace::global, const std::string& dtype = {""});
  // static mlir::Value build(ComputeDAG* graph, mlir::Value A, float B, std::string operation, MemorySpace ms, const std::string& dtype = {""});
  static mlir::Value add(mlir::OpBuilder &builder, mlir::Value elem_1, mlir::Value elem_2);
  static mlir::Value mul(mlir::OpBuilder &builder, mlir::Value elem_1, mlir::Value elem_2);
  static mlir::Value div(mlir::OpBuilder &builder, mlir::Value elem_1, mlir::Value elem_2);
  static mlir::Value sub(mlir::OpBuilder &builder, mlir::Value elem_1, mlir::Value elem_2);
  static mlir::Value pow(mlir::OpBuilder &builder, mlir::Value elem_1, mlir::Value elem_2);
  static mlir::Value equal(mlir::OpBuilder &builder, mlir::Value elem_1, mlir::Value elem_2);
  static mlir::Value greater(mlir::OpBuilder &builder, mlir::Value elem_1, mlir::Value elem_2);
  static std::map<std::string, std::function<mlir::Value(mlir::OpBuilder&, mlir::Value, mlir::Value)>> operationMap;
};

struct ElementWise : Operator<ElementWise> {
  static mlir::Value build(ComputeDAG* graph, mlir::Value input, std::string operation, MemorySpace ms, const std::string& dtype = {""});
  static mlir::Value tanh(mlir::OpBuilder &builder, mlir::Value elem, mlir::Type type);
  static mlir::Value sqrt(mlir::OpBuilder &builder, mlir::Value elem, mlir::Type type);
  static mlir::Value log(mlir::OpBuilder &builder, mlir::Value elem, mlir::Type type);
  static mlir::Value relu(mlir::OpBuilder &builder, mlir::Value elem, mlir::Type type);
  static mlir::Value cast(mlir::OpBuilder &builder, mlir::Value elem, mlir::Type type);
  static mlir::Value gelu(mlir::OpBuilder &builder, mlir::Value elem, mlir::Type type);
  static std::map<std::string, std::function<mlir::Value(mlir::OpBuilder&, mlir::Value, mlir::Type)>> operationMap;
};

struct LayerNorm : Operator<LayerNorm> {
  static mlir::Value build(ComputeDAG* graph, mlir::Value input, mlir::Value scale, mlir::Value bias, int64_t axis, 
  const float &eps=1e-5, MemorySpace ms=MemorySpace::global, const std::string& dtype_ = {""});
};

struct Gather : Operator<Gather> {
  static mlir::Value build(ComputeDAG* graph, mlir::Value input, mlir::Value indices, const std::int64_t& axis=0, const std::string& dtype_ = {""});
  // static std::vector<mlir::Value> build(ComputeDAG* graph, mlir::Value input, int indices, const std::int64_t& axis=0, const std::string& dtype_ = {""});
};

}
