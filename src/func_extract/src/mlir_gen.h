#ifndef FUNC_EXTRACT_MLIR_GEN_H
#define FUNC_EXTRACT_MLIR_GEN_H

// IMPORTANT: MLIR headers must come BEFORE check_regs.h because check_regs.h
// defines macros (#define context, #define builder) that clash with MLIR headers.
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "global_data_struct.h"
#include "check_regs.h"
#include "ast.h"
#include "ins_context_stack.h"

// Undefine the macros from check_regs.h that clash with normal code
#undef context
#undef builder

#include <string>
#include <map>
#include <memory>
#include <cstdint>

namespace funcExtract {

/// MLIRUpdateFunctionGen: an alternative code generator that emits MLIR
/// in the Arithmetic/SCF/MemRef/Affine dialects instead of raw LLVM IR.
/// This preserves structured control flow (loops, if/else) which is essential
/// for the TensorLift semantic lifting stage.
class MLIRUpdateFunctionGen : public UFGenerator {
public:
  MLIRUpdateFunctionGen();
  ~MLIRUpdateFunctionGen() override;

  void print_llvm_ir(DestInfo &destInfo,
                     const uint32_t bound,
                     uint32_t instrIdx,
                     std::string fileName) override;

private:
  // MLIR infrastructure
  mlir::MLIRContext mlirCtx;
  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;
  mlir::FuncOp theFunction;
  mlir::Block *entryBlock;

  // Mirrors of UpdateFunctionGen state
  HierCtx insContextStk;
  uint32_t cct_cnt;
  uint32_t ext_cnt;
  std::map<std::string, uint32_t> cctNameIdxMap;
  std::map<std::string, uint32_t> extNameIdxMap;
  std::vector<std::pair<std::string, std::string>> memInstances;
  struct InstrInfo_t currInstrInfo;
  bool ignoreSubModules;
  bool skipCheck;
  std::map<astNode*, uint32_t> CLEAN_QUEUE;
  std::map<astNode*, uint32_t> DIRTY_QUEUE;
  std::set<std::string> resetedReg;

  // Maps for tracking named values (replacing LLVM's ValueSymbolTable)
  std::map<std::string, mlir::Value> namedValues;
  // Maps for function arguments by name
  std::map<std::string, mlir::Value> funcArgMap;
  // Dynamic data per module (mirrors UpdateFunctionGen)
  std::map<std::string, std::shared_ptr<ModuleDynInfo_t>> dynDataMp;
  std::map<std::string, astNode*> memNodeMap;

  // --- Core constraint-to-MLIR translation functions ---

  /// Main dispatch: translate a variable name to an MLIR value
  mlir::Value add_constraint(std::string varAndSlice, uint32_t timeIdx,
                             const uint32_t bound);

  /// Main dispatch: translate an AST node to an MLIR value
  mlir::Value add_constraint(astNode* const node, uint32_t timeIdx,
                             const uint32_t bound);

  /// Handle nonblocking assignment (register update across time steps)
  mlir::Value add_nb_constraint(astNode* const node, uint32_t timeIdx,
                                const uint32_t bound);

  /// Handle SSA-like assignments (wire-type dispatch by node type)
  mlir::Value add_ssa_constraint(astNode* const node, uint32_t timeIdx,
                                 const uint32_t bound);

  // --- Constraint type handlers (mirrors op_constraint.cpp) ---

  /// Two-operand ops: add, sub, mul, and, or, xor, shifts, comparisons
  mlir::Value two_op_constraint(astNode* const node, uint32_t timeIdx,
                                const uint32_t bound);

  /// Single-operand ops: not, negate
  mlir::Value one_op_constraint(astNode* const node, uint32_t timeIdx,
                                const uint32_t bound);

  /// Reduction operations (e.g., |a, &a, ^a)
  mlir::Value reduce_one_op_constraint(astNode* const node, uint32_t timeIdx,
                                       const uint32_t bound);

  /// ITE (if-then-else mux): condition ? true_val : false_val
  mlir::Value ite_op_constraint(astNode* const node, uint32_t timeIdx,
                                const uint32_t bound);

  /// Case statement (chain of selects)
  mlir::Value case_constraint(astNode* const node, uint32_t timeIdx,
                              const uint32_t bound);

  /// Switch with lookup table
  mlir::Value switch_constraint(astNode* const node, uint32_t timeIdx,
                                const uint32_t bound);

  /// Bit selection: signal[hi:lo]
  mlir::Value sel_op_constraint(astNode* const node, uint32_t timeIdx,
                                const uint32_t bound);

  /// Concatenation: {a, b, c}
  mlir::Value src_concat_op_constraint(astNode* const node, uint32_t timeIdx,
                                       const uint32_t bound);

  /// Numeric literal
  mlir::Value num_constraint(astNode* const node, uint32_t timeIdx);

  /// Input/argument reference
  mlir::Value input_constraint(astNode* const node, uint32_t timeIdx,
                               const uint32_t bound);

  /// Dynamic memory read (array[index])
  mlir::Value dyn_sel_constraint(astNode* const node, uint32_t timeIdx,
                                 const uint32_t bound);

  /// Memory write (conditional store)
  void mem_assign_constraint(astNode* const node, uint32_t timeIdx,
                             const uint32_t bound);

  /// Sub-module function call
  mlir::Value submod_constraint(astNode* const node, uint32_t timeIdx,
                                const uint32_t bound);

  // --- Arithmetic helper functions ---

  /// Create an MLIR arithmetic instruction from an operator string
  mlir::Value make_mlir_instr(const std::string &op,
                              mlir::Value op1, mlir::Value op2,
                              uint32_t destWidth,
                              uint32_t op1Width, uint32_t op2Width,
                              bool isSigned = false);

  /// Bit extraction: extract bits [high:low] from a value
  mlir::Value extract_bits(mlir::Value in, uint32_t high, uint32_t low);

  /// Concatenate two values: {val1, val2}
  mlir::Value concat_values(mlir::Value val1, mlir::Value val2);

  /// Zero-extend a value to the given width
  mlir::Value zext(mlir::Value v, uint32_t targetWidth);

  /// Sign-extend a value to the given width
  mlir::Value sext(mlir::Value v, uint32_t targetWidth);

  /// Truncate a value to the given width
  mlir::Value trunc(mlir::Value v, uint32_t targetWidth);

  /// Create an integer constant
  mlir::Value constant(uint64_t val, uint32_t width);

  /// Parse a binary/hex/decimal string literal to a constant
  mlir::Value literal_to_constant(const std::string &lit, uint32_t width);

  // --- Utility functions ---

  /// Get the integer bit-width of an mlir::Value
  uint32_t getWidth(mlir::Value v);

  /// Look up a named value (argument or previously computed)
  mlir::Value lookupValue(const std::string &name);

  /// Register a named value
  void registerValue(const std::string &name, mlir::Value v);

  /// Get dynamic data for a module
  std::shared_ptr<ModuleDynInfo_t>
  get_dyn_data(std::shared_ptr<ModuleInfo_t> curMod);

  /// Clean all module dynamic info
  void clean_all_mod_dynamic_info();

  void clean_data();

  void push_clean_queue(astNode* node, uint32_t timeIdx);
  void push_dirty_queue(astNode* node, uint32_t timeIdx);
  bool is_in_clean_queue(std::string var);
  bool is_in_dirty_queue(std::string var);

  void initialize_min_delay(std::shared_ptr<ModuleInfo_t> &modInfo,
                            std::string outPort);

  /// Get the MLIR location for generated ops
  mlir::Location getLoc();
};

} // namespace funcExtract

#endif // FUNC_EXTRACT_MLIR_GEN_H
