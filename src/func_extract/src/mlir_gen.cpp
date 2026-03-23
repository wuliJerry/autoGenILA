// mlir_gen.cpp — MLIR-based update function generator for TensorLift Path A
//
// This implements MLIRUpdateFunctionGen, which emits MLIR in the
// Arithmetic/SCF/MemRef/Affine dialects instead of raw LLVM IR.
// The dispatch logic mirrors check_regs.cpp and op_constraint.cpp.

#include "mlir_gen.h"
#include "parse_fill.h"
#include "helper.h"
#include "util.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"

#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace taintGen;

namespace funcExtract {

extern std::string DELIM;

// ============================================================================
// Construction / Destruction
// ============================================================================

MLIRUpdateFunctionGen::MLIRUpdateFunctionGen()
    : builder(&mlirCtx), entryBlock(nullptr),
      cct_cnt(0), ext_cnt(0),
      ignoreSubModules(false), skipCheck(true) {
  mlirCtx.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
  mlirCtx.getOrLoadDialect<mlir::scf::SCFDialect>();
  mlirCtx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  mlirCtx.getOrLoadDialect<mlir::AffineDialect>();
  mlirCtx.getOrLoadDialect<mlir::StandardOpsDialect>();
}

MLIRUpdateFunctionGen::~MLIRUpdateFunctionGen() = default;

// ============================================================================
// Utility helpers
// ============================================================================

mlir::Location MLIRUpdateFunctionGen::getLoc() {
  return builder.getUnknownLoc();
}

uint32_t MLIRUpdateFunctionGen::getWidth(mlir::Value v) {
  if (auto intTy = v.getType().dyn_cast<mlir::IntegerType>())
    return intTy.getWidth();
  assert(false && "Expected integer type");
  return 0;
}

mlir::Value MLIRUpdateFunctionGen::lookupValue(const std::string &name) {
  auto it = namedValues.find(name);
  if (it != namedValues.end()) return it->second;
  auto it2 = funcArgMap.find(name);
  if (it2 != funcArgMap.end()) return it2->second;
  return mlir::Value();
}

void MLIRUpdateFunctionGen::registerValue(const std::string &name, mlir::Value v) {
  namedValues[name] = v;
}

mlir::Value MLIRUpdateFunctionGen::constant(uint64_t val, uint32_t width) {
  auto ty = builder.getIntegerType(width);
  return builder.create<mlir::arith::ConstantIntOp>(getLoc(),
      static_cast<int64_t>(val), ty);
}

mlir::Value MLIRUpdateFunctionGen::zext(mlir::Value v, uint32_t targetWidth) {
  uint32_t curWidth = getWidth(v);
  if (curWidth == targetWidth) return v;
  if (curWidth > targetWidth) return trunc(v, targetWidth);
  auto destTy = builder.getIntegerType(targetWidth);
  return builder.create<mlir::arith::ExtUIOp>(getLoc(), destTy, v);
}

mlir::Value MLIRUpdateFunctionGen::sext(mlir::Value v, uint32_t targetWidth) {
  uint32_t curWidth = getWidth(v);
  if (curWidth == targetWidth) return v;
  if (curWidth > targetWidth) return trunc(v, targetWidth);
  auto destTy = builder.getIntegerType(targetWidth);
  return builder.create<mlir::arith::ExtSIOp>(getLoc(), destTy, v);
}

mlir::Value MLIRUpdateFunctionGen::trunc(mlir::Value v, uint32_t targetWidth) {
  uint32_t curWidth = getWidth(v);
  if (curWidth == targetWidth) return v;
  assert(curWidth > targetWidth);
  auto destTy = builder.getIntegerType(targetWidth);
  return builder.create<mlir::arith::TruncIOp>(getLoc(), destTy, v);
}

mlir::Value MLIRUpdateFunctionGen::extract_bits(mlir::Value in,
                                                 uint32_t high, uint32_t low) {
  uint32_t inWidth = getWidth(in);
  uint32_t outWidth = high - low + 1;
  if (low == 0 && outWidth == inWidth) return in;

  mlir::Value result = in;
  if (low > 0) {
    mlir::Value shamt = constant(low, inWidth);
    result = builder.create<mlir::arith::ShRUIOp>(getLoc(), result, shamt);
  }
  if (outWidth < inWidth)
    result = trunc(result, outWidth);
  return result;
}

mlir::Value MLIRUpdateFunctionGen::concat_values(mlir::Value val1,
                                                  mlir::Value val2) {
  uint32_t w1 = getWidth(val1);
  uint32_t w2 = getWidth(val2);
  uint32_t totalWidth = w1 + w2;

  mlir::Value ext1 = zext(val1, totalWidth);
  mlir::Value ext2 = zext(val2, totalWidth);
  mlir::Value shamt = constant(w2, totalWidth);
  mlir::Value shifted = builder.create<mlir::arith::ShLIOp>(getLoc(), ext1, shamt);
  return builder.create<mlir::arith::OrIOp>(getLoc(), shifted, ext2);
}

mlir::Value MLIRUpdateFunctionGen::literal_to_constant(
    const std::string &lit, uint32_t width) {
  uint64_t val = 0;
  std::string s = lit;

  auto tickPos = s.find('\'');
  if (tickPos != std::string::npos) {
    char base = s[tickPos + 1];
    std::string digits = s.substr(tickPos + 2);
    digits.erase(std::remove(digits.begin(), digits.end(), '_'), digits.end());

    try {
      if (base == 'h' || base == 'H') {
        for (auto &c : digits)
          if (c == 'x' || c == 'X' || c == 'z' || c == 'Z') c = '0';
        if (!digits.empty()) val = std::stoull(digits, nullptr, 16);
      } else if (base == 'b' || base == 'B') {
        for (auto &c : digits)
          if (c == 'x' || c == 'X' || c == 'z' || c == 'Z') c = '0';
        if (!digits.empty()) val = std::stoull(digits, nullptr, 2);
      } else if (base == 'd' || base == 'D') {
        if (!digits.empty()) val = std::stoull(digits, nullptr, 10);
      } else if (base == 'o' || base == 'O') {
        if (!digits.empty()) val = std::stoull(digits, nullptr, 8);
      }
    } catch (...) {
      toCoutVerb("MLIR literal_to_constant: parse error for: " + lit);
      val = 0;
    }
  } else {
    // Handle hex strings without tick (e.g., "ff", "0x1a")
    try {
      if (s.find("0x") == 0 || s.find("0X") == 0)
        val = std::stoull(s, nullptr, 16);
      else
        val = std::stoull(s, nullptr, 10);
    } catch (...) {
      toCoutVerb("MLIR literal_to_constant: could not parse: " + lit);
      val = 0;
    }
  }

  if (width < 64) val &= (1ULL << width) - 1;
  return constant(val, width);
}

// ============================================================================
// make_mlir_instr: translate operator string to MLIR arithmetic op
// ============================================================================

mlir::Value MLIRUpdateFunctionGen::make_mlir_instr(
    const std::string &op, mlir::Value op1, mlir::Value op2,
    uint32_t destWidth, uint32_t op1Width, uint32_t op2Width,
    bool isSigned) {

  uint32_t maxWidth = std::max({destWidth, op1Width, op2Width});
  if (getWidth(op1) < maxWidth) op1 = isSigned ? sext(op1, maxWidth) : zext(op1, maxWidth);
  if (getWidth(op2) < maxWidth) op2 = isSigned ? sext(op2, maxWidth) : zext(op2, maxWidth);

  mlir::Value result;
  auto loc = getLoc();

  if      (op == "+" || op == "add") result = builder.create<mlir::arith::AddIOp>(loc, op1, op2);
  else if (op == "-" || op == "sub") result = builder.create<mlir::arith::SubIOp>(loc, op1, op2);
  else if (op == "*" || op == "mul") result = builder.create<mlir::arith::MulIOp>(loc, op1, op2);
  else if (op == "&" || op == "and") result = builder.create<mlir::arith::AndIOp>(loc, op1, op2);
  else if (op == "|" || op == "or")  result = builder.create<mlir::arith::OrIOp>(loc, op1, op2);
  else if (op == "^" || op == "xor") result = builder.create<mlir::arith::XOrIOp>(loc, op1, op2);
  else if (op == "<<" || op == "shl") result = builder.create<mlir::arith::ShLIOp>(loc, op1, op2);
  else if (op == ">>" || op == "srl") result = builder.create<mlir::arith::ShRUIOp>(loc, op1, op2);
  else if (op == ">>>" || op == "sra") result = builder.create<mlir::arith::ShRSIOp>(loc, op1, op2);
  else if (op == "==" || op == "eq")
    result = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, op1, op2);
  else if (op == "!=" || op == "ne")
    result = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, op1, op2);
  else if (op == "<")
    result = builder.create<mlir::arith::CmpIOp>(loc,
        isSigned ? mlir::arith::CmpIPredicate::slt : mlir::arith::CmpIPredicate::ult, op1, op2);
  else if (op == "<=")
    result = builder.create<mlir::arith::CmpIOp>(loc,
        isSigned ? mlir::arith::CmpIPredicate::sle : mlir::arith::CmpIPredicate::ule, op1, op2);
  else if (op == ">")
    result = builder.create<mlir::arith::CmpIOp>(loc,
        isSigned ? mlir::arith::CmpIPredicate::sgt : mlir::arith::CmpIPredicate::ugt, op1, op2);
  else if (op == ">=")
    result = builder.create<mlir::arith::CmpIOp>(loc,
        isSigned ? mlir::arith::CmpIPredicate::sge : mlir::arith::CmpIPredicate::uge, op1, op2);
  else if (op == "&&") {
    // Logical AND: convert both to i1 (non-zero check), then AND
    auto i1Ty = builder.getIntegerType(1);
    mlir::Value zero1 = builder.create<mlir::arith::ConstantIntOp>(loc, 0, op1.getType());
    mlir::Value zero2 = builder.create<mlir::arith::ConstantIntOp>(loc, 0, op2.getType());
    mlir::Value lhs = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, op1, zero1);
    mlir::Value rhs = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, op2, zero2);
    result = builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
  }
  else if (op == "||") {
    auto i1Ty = builder.getIntegerType(1);
    mlir::Value zero1 = builder.create<mlir::arith::ConstantIntOp>(loc, 0, op1.getType());
    mlir::Value zero2 = builder.create<mlir::arith::ConstantIntOp>(loc, 0, op2.getType());
    mlir::Value lhs = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, op1, zero1);
    mlir::Value rhs = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, op2, zero2);
    result = builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
  }
  else { toCout("MLIR gen: unsupported op: " + op); abort(); }

  if (result && getWidth(result) > destWidth)
    result = trunc(result, destWidth);
  return result;
}

// ============================================================================
// print_llvm_ir: Main entry point — builds the MLIR module
// ============================================================================

void MLIRUpdateFunctionGen::print_llvm_ir(
    DestInfo &destInfo, const uint32_t bound,
    uint32_t instrIdx, std::string fileName) {

  std::string destName = destInfo.get_dest_name();
  std::string curModName = destInfo.get_mod_name();
  std::string curInsName = destInfo.get_ins_name();
  auto curMod = g_moduleInfoMap[curModName];
  auto curDynData = get_dyn_data(curMod);

  std::string insName = curModName == g_topModule ? curModName : curInsName;
  Context_t insCntxt(insName, "", curMod, nullptr, nullptr);
  insContextStk.clear();
  clean_all_mod_dynamic_info();
  namedValues.clear();
  funcArgMap.clear();
  cct_cnt = 0;
  ext_cnt = 0;

  if (curModName == g_topModule) {
    insContextStk.push_back(insCntxt);
  } else {
    while (curMod->name != g_topModule) {
      curDynData->isFunctionedSubMod = false;
      auto parentMod = *(curMod->parentModVec.begin());
      insName = ask_parent_my_ins_name(curMod->name, parentMod);
      Context_t ctx(insName, "", curMod, parentMod, nullptr);
      insContextStk.insert(insContextStk.begin(), ctx);
      curMod = parentMod;
      curDynData = get_dyn_data(curMod);
    }
    Context_t ctx(curMod->name, "", curMod, nullptr, nullptr);
    curDynData->isFunctionedSubMod = true;
    insContextStk.insert(insContextStk.begin(), ctx);
  }

  std::string destPrefix = insContextStk.get_hier_name(false);
  if (!destPrefix.empty()) destName = destPrefix + "." + destName;

  theModule = mlir::ModuleOp::create(getLoc(),
      llvm::StringRef("mod_;_" + curMod->name + "_;_" + destName));

  std::vector<std::string> destVec = destInfo.get_no_slice_name();
  std::shared_ptr<ModuleInfo_t> topModInfo = g_moduleInfoMap[g_topModule];

  RegWidthVec_t regWidth;
  collect_regs(topModInfo, "", regWidth);

  // Build function argument types
  std::vector<mlir::Type> argTypes;
  std::vector<std::string> argNames;

  // 1) Scalar register args
  for (auto &w : regWidth) {
    if (get_vector_of_target(w.first).empty()) {
      argTypes.push_back(builder.getIntegerType(w.second));
      argNames.push_back(w.first + post_fix(bound));
    }
  }

  // 2) Register array args → memref
  for (auto &pair : g_allowedTgtVec) {
    uint32_t width = get_var_slice_width_simp(pair.second.members[0], curMod);
    uint32_t paddedWidth = get_padded_width(width);
    uint32_t arraySize = pair.second.members.size();
    auto memrefTy = mlir::MemRefType::get(
        {static_cast<int64_t>(arraySize)}, builder.getIntegerType(paddedWidth));
    argTypes.push_back(memrefTy);
    argNames.push_back(pair.first + post_fix(bound));
  }

  // 3) Memory module output args
  collect_mem_ins(topModInfo, "", memInstances);
  for (auto &inst : memInstances) {
    auto memMod = g_moduleInfoMap[inst.second];
    for (uint32_t i = 0; i <= bound; i++)
      for (auto &output : memMod->moduleOutputs) {
        uint32_t width = get_var_slice_width_simp(output, memMod);
        argTypes.push_back(builder.getIntegerType(width));
        argNames.push_back(inst.first + "." + output + post_fix(i));
      }
  }

  // 4) Module input args — grouped as memref<(bound+1) x width> per input signal
  // This enables indexed access by time step, which is essential for loop structuring.
  // Each input signal gets one memref argument instead of (bound+1) scalar args.
  std::vector<std::string> moduleInputNames; // track for later load emission
  for (auto &inp : curMod->moduleInputs) {
    if (inp == curMod->clk) continue;
    uint32_t width = get_var_slice_width_simp(inp, curMod);
    auto elemTy = builder.getIntegerType(width);
    auto memrefTy = mlir::MemRefType::get({static_cast<int64_t>(bound + 1)}, elemTy);
    argTypes.push_back(memrefTy);
    argNames.push_back(inp + "_inputs");
    moduleInputNames.push_back(inp);
  }

  // 5) Return array arg (if vector target)
  bool isVectorDest = destInfo.isVector && !destInfo.isMemVec;
  if (isVectorDest) {
    uint32_t width = get_var_slice_width_simp(destVec[0], curMod);
    uint32_t paddedWidth = get_padded_width(width);
    auto memrefTy = mlir::MemRefType::get(
        {static_cast<int64_t>(destVec.size())}, builder.getIntegerType(paddedWidth));
    argTypes.push_back(memrefTy);
    argNames.push_back(RETURN_ARRAY_PTR_ID);
  }

  // Return type
  std::vector<mlir::Type> resultTypes;
  if (!destInfo.isVector) {
    uint32_t retWidth = get_var_slice_width_simp(destVec.front(), curMod);
    resultTypes.push_back(builder.getIntegerType(retWidth));
  }

  auto funcType = builder.getFunctionType(argTypes, resultTypes);
  std::string funcName = destInfo.get_func_name();

  builder.setInsertionPointToEnd(theModule.getBody());
  theFunction = builder.create<mlir::FuncOp>(getLoc(), funcName, funcType);

  entryBlock = theFunction.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  // Register argument names
  for (uint32_t i = 0; i < argNames.size(); i++)
    funcArgMap[argNames[i]] = entryBlock->getArgument(i);

  // Load register array elements
  for (auto &pair : g_allowedTgtVec) {
    std::string arrayNameBound = pair.first + post_fix(bound);
    mlir::Value arrayArg = funcArgMap[arrayNameBound];
    uint32_t width = get_var_slice_width_simp(pair.second.members[0], curMod);
    uint32_t paddedWidth = get_padded_width(width);

    int idx = 0;
    for (const std::string &member : pair.second.members) {
      std::string memberNameBound = member + post_fix(bound);
      mlir::Value idxVal = builder.create<mlir::arith::ConstantIndexOp>(getLoc(), idx);
      mlir::Value loaded = builder.create<mlir::memref::LoadOp>(getLoc(), arrayArg,
          mlir::ValueRange{idxVal});
      if (paddedWidth != width) loaded = trunc(loaded, width);
      registerValue(memberNameBound, loaded);
      idx++;
    }
  }

  // Emit loads for module inputs from memref args (for each time step)
  for (auto &inp : moduleInputNames) {
    std::string memrefArgName = inp + "_inputs";
    mlir::Value memrefArg = funcArgMap[memrefArgName];
    for (uint32_t t = 0; t <= bound; t++) {
      std::string timedName = inp + post_fix(t);
      mlir::Value tIdx = builder.create<mlir::arith::ConstantIndexOp>(getLoc(), t);
      mlir::Value loaded = builder.create<mlir::memref::LoadOp>(getLoc(), memrefArg,
          mlir::ValueRange{tIdx});
      registerValue(timedName, loaded);
    }
  }

  // Generate update function body
  ignoreSubModules = false;
  skipCheck = true;
  clean_data();
  curMod = g_moduleInfoMap[curModName];

  if (!destInfo.isVector) {
    if (is_output(destName, curMod)) initialize_min_delay(curMod, destName);
    std::string dest = destVec.front();
    if (curMod->visitedNode.find(dest) == curMod->visitedNode.end()
        && curMod->reg2Slices.find(dest) == curMod->reg2Slices.end()) {
      toCout("MLIR Error: ast node not found for: |" + dest + "|");
      abort();
    }
    mlir::Value destExpr = add_constraint(dest, 0, bound);
    builder.create<mlir::ReturnOp>(getLoc(), mlir::ValueRange{destExpr});
  } else {
    std::vector<mlir::Value> retVec;
    std::string modName = destInfo.get_mod_name();
    std::string insNameV = destInfo.get_ins_name();
    if (insNameV.empty()) insNameV = modName;

    for (std::string &dest : destVec) {
      if (is_output(dest, curMod)) initialize_min_delay(curMod, dest);
      curMod = g_moduleInfoMap[modName];
      assert(modName == g_topModule);
      insContextStk.clear();
      Context_t ctx(insNameV, dest, curMod, nullptr, nullptr);
      insContextStk.push_back(ctx);

      if (curMod->visitedNode.find(dest) == curMod->visitedNode.end()
          && curMod->reg2Slices.find(dest) == curMod->reg2Slices.end()) {
        toCout("MLIR Error: ast node not found for: |" + dest + "|");
        abort();
      }
      mlir::Value destExpr = add_constraint(dest, 0, bound);
      if (!destInfo.isMemVec) retVec.push_back(destExpr);
    }

    if (!retVec.empty()) {
      mlir::Value retArray = funcArgMap[RETURN_ARRAY_PTR_ID];
      uint32_t w = get_var_slice_width_simp(destVec[0], curMod);
      uint32_t paddedWidth = get_padded_width(w);
      for (uint32_t i = 0; i < retVec.size(); i++) {
        mlir::Value idxVal = builder.create<mlir::arith::ConstantIndexOp>(getLoc(), i);
        mlir::Value val = retVec[i];
        if (getWidth(val) < paddedWidth) val = zext(val, paddedWidth);
        else if (getWidth(val) > paddedWidth) val = trunc(val, paddedWidth);
        builder.create<mlir::memref::StoreOp>(getLoc(), val, retArray,
            mlir::ValueRange{idxVal});
      }
    }
    builder.create<mlir::ReturnOp>(getLoc(), mlir::ValueRange{});
  }

  // Verify and output
  if (mlir::failed(mlir::verify(theModule))) {
    toCout("MLIR module verification failed!");
    theModule.dump();
    abort();
  }

  // Write MLIR output file
  std::string mlirFileName = fileName;
  auto pos = mlirFileName.rfind("-ll");
  if (pos != std::string::npos) mlirFileName.replace(pos, 3, ".mlir");
  else { pos = mlirFileName.rfind(".ll"); if (pos != std::string::npos) mlirFileName.replace(pos, 3, ".mlir"); else mlirFileName += ".mlir"; }

  std::string mlirStr;
  llvm::raw_string_ostream os(mlirStr);
  theModule.print(os);
  os.flush();

  std::ofstream output(mlirFileName);
  output << mlirStr << std::endl;
  output.close();

  // Write stub to original filename for pipeline compatibility
  std::ofstream output2(fileName);
  output2 << "; MLIR output written to: " << mlirFileName << std::endl;
  output2.close();

  toCout("** Finish MLIR update function for: " + destName);
  toCout("   Written to: " + mlirFileName);
}

// ============================================================================
// add_constraint: by variable name
// ============================================================================

mlir::Value MLIRUpdateFunctionGen::add_constraint(
    std::string varAndSlice, uint32_t timeIdx, const uint32_t bound) {

  auto curMod = insContextStk.get_curMod();
  std::string var, varSlice;
  split_slice(varAndSlice, var, varSlice);

  if (curMod->reg2Slices.find(var) == curMod->reg2Slices.end()) {
    if (curMod->visitedNode.find(var) == curMod->visitedNode.end()) {
      toCout("MLIR gen error: cannot find node for: " + varAndSlice);
      abort();
    }
    return add_constraint(curMod->visitedNode[var], timeIdx, bound);
  } else {
    mlir::Value ret;
    bool first = true;
    for (std::string slice : curMod->reg2Slices[var]) {
      if (curMod->visitedNode.find(slice) == curMod->visitedNode.end()) {
        toCout("MLIR gen error: cannot find node for slice: " + slice);
        abort();
      }
      mlir::Value tmpSlice = add_constraint(curMod->visitedNode[slice], timeIdx, bound);
      if (first) { ret = tmpSlice; first = false; }
      else ret = concat_values(ret, tmpSlice);
    }
    return ret;
  }
}

// ============================================================================
// add_constraint: by AST node — main predicate-based dispatch
// ============================================================================

mlir::Value MLIRUpdateFunctionGen::add_constraint(
    astNode* const node, uint32_t timeIdx, const uint32_t bound) {

  if (!node) return mlir::Value();

  std::string varAndSlice = node->dest;

  // "default" is a case label, not a real variable — dispatch based on type only
  if (varAndSlice == "default") {
    return add_ssa_constraint(node, timeIdx, bound);
  }

  auto curMod = insContextStk.get_curMod();
  auto curDynData = get_dyn_data(curMod);

  toCoutVerb("MLIR add_constraint for: " + varAndSlice
             + ", t:" + std::to_string(timeIdx) + ", b:" + std::to_string(bound));

  if (timeIdx > bound) {
    uint32_t width = insContextStk.get_var_slice_width_simp(varAndSlice);
    return constant(0, width);
  }

  // Check cache
  std::string timedName = timed_name(varAndSlice, timeIdx);
  mlir::Value cached = lookupValue(timedName);
  if (cached) return cached;

  mlir::Value retExpr;

  if (is_input(varAndSlice, curMod))
    retExpr = input_constraint(node, timeIdx, bound);
  else if (is_reg_in_curMod(varAndSlice, curMod)
           && node->type != SRC_CONCAT && node->type != SWITCH)
    retExpr = add_nb_constraint(node, timeIdx, bound);
  else if (is_number(varAndSlice))
    retExpr = num_constraint(node, timeIdx);
  else if (is_case_dest(varAndSlice, curMod))
    retExpr = case_constraint(node, timeIdx, bound);
  else if (is_switch_dest(varAndSlice, curMod))
    retExpr = switch_constraint(node, timeIdx, bound);
  else if (is_submod_output(varAndSlice, curMod))
    retExpr = submod_constraint(node, timeIdx, bound);
  else if (node->type == MEM_IF_ASSIGN) {
    mem_assign_constraint(node, timeIdx, bound);
    return mlir::Value();
  }
  else if (node->type == DYNSEL)
    retExpr = dyn_sel_constraint(node, timeIdx, bound);
  else
    retExpr = add_ssa_constraint(node, timeIdx, bound);

  if (retExpr) registerValue(timedName, retExpr);
  return retExpr;
}

// ============================================================================
// add_ssa_constraint: wire-type dispatch
// ============================================================================

mlir::Value MLIRUpdateFunctionGen::add_ssa_constraint(
    astNode* const node, uint32_t timeIdx, const uint32_t bound) {

  toCoutVerb("MLIR ssa_constraint for: " + node->dest + " t:" + std::to_string(timeIdx));

  switch (node->type) {
    case TWO_OP:     return two_op_constraint(node, timeIdx, bound);
    case ONE_OP:     return one_op_constraint(node, timeIdx, bound);
    case REDUCE1:    return reduce_one_op_constraint(node, timeIdx, bound);
    case SEL:        return sel_op_constraint(node, timeIdx, bound);
    case SRC_CONCAT: return src_concat_op_constraint(node, timeIdx, bound);
    case ITE:        return ite_op_constraint(node, timeIdx, bound);
    default:
      toCout("MLIR gen error: no ssa_constraint for: " + node->dest
             + ", type:" + std::to_string(node->type));
      abort();
      return mlir::Value();
  }
}

// ============================================================================
// Constraint handlers
// ============================================================================

mlir::Value MLIRUpdateFunctionGen::two_op_constraint(
    astNode* const node, uint32_t timeIdx, const uint32_t bound) {

  toCoutVerb("MLIR two_op for: " + node->dest);
  auto curMod = insContextStk.get_curMod();

  assert(node->srcVec.size() == 2);
  std::string destAndSlice = node->dest;
  std::string op1AndSlice = node->srcVec[0];
  std::string op2AndSlice = node->srcVec[1];
  std::string op1, op1Slice, op2, op2Slice;
  split_slice(op1AndSlice, op1, op1Slice);
  split_slice(op2AndSlice, op2, op2Slice);

  uint32_t destWidthNum = get_var_slice_width_simp(destAndSlice, curMod);
  uint32_t op1WidthNum = get_var_slice_width_simp(op1AndSlice, curMod);
  uint32_t op2WidthNum = get_var_slice_width_simp(op2AndSlice, curMod);

  mlir::Value op1Expr;
  if (!is_number(op1)) {
    mlir::Value tmp = add_constraint(node->childVec[0], timeIdx, bound);
    if (op1Slice.empty() || has_direct_assignment(op1AndSlice, curMod))
      op1Expr = tmp;
    else {
      uint32_t hi = get_lgc_hi(op1AndSlice, curMod);
      uint32_t lo = get_lgc_lo(op1AndSlice, curMod);
      op1Expr = extract_bits(tmp, hi, lo);
    }
  } else {
    op1Expr = literal_to_constant(op1AndSlice, op1WidthNum);
  }

  mlir::Value op2Expr;
  if (!is_number(op2)) {
    mlir::Value tmp = add_constraint(node->childVec[1], timeIdx, bound);
    if (op2Slice.empty() || has_direct_assignment(op2AndSlice, curMod))
      op2Expr = tmp;
    else {
      uint32_t hi = get_lgc_hi(op2AndSlice, curMod);
      uint32_t lo = get_lgc_lo(op2AndSlice, curMod);
      op2Expr = extract_bits(tmp, hi, lo);
    }
  } else {
    op2Expr = literal_to_constant(op2AndSlice, op2WidthNum);
  }

  if (!op1Expr || !op2Expr) {
    toCout("MLIR gen: null operand in two_op for: " + destAndSlice);
    return mlir::Value();
  }

  bool isSigned = !node->extVec.empty() && (node->extVec[0] == 1 || node->extVec[1] == 1);
  return make_mlir_instr(node->op, op1Expr, op2Expr,
                         destWidthNum, getWidth(op1Expr), getWidth(op2Expr), isSigned);
}

mlir::Value MLIRUpdateFunctionGen::one_op_constraint(
    astNode* const node, uint32_t timeIdx, const uint32_t bound) {

  toCoutVerb("MLIR one_op for: " + node->dest);
  auto curMod = insContextStk.get_curMod();
  assert(node->srcVec.size() == 1);

  std::string op1AndSlice = node->srcVec[0];
  std::string op1, op1Slice;
  split_slice(op1AndSlice, op1, op1Slice);

  mlir::Value operand;
  if (is_number(op1)) {
    uint32_t w = get_var_slice_width_simp(op1AndSlice, curMod);
    operand = literal_to_constant(op1AndSlice, w);
  } else {
    mlir::Value tmp = add_constraint(node->childVec[0], timeIdx, bound);
    if (op1Slice.empty() || has_direct_assignment(op1AndSlice, curMod))
      operand = tmp;
    else {
      uint32_t hi = get_lgc_hi(op1AndSlice, curMod);
      uint32_t lo = get_lgc_lo(op1AndSlice, curMod);
      operand = extract_bits(tmp, hi, lo);
    }
  }
  if (!operand) return mlir::Value();

  uint32_t width = getWidth(operand);
  std::string op = node->op;

  if (op.empty()) {
    // Identity/passthrough: assign X = Y; (no operator)
    return operand;
  } else if (op == "~" || op == "not") {
    mlir::Value allOnes = constant(width < 64 ? (1ULL << width) - 1 : ~0ULL, width);
    return builder.create<mlir::arith::XOrIOp>(getLoc(), operand, allOnes);
  } else if (op == "-" || op == "neg") {
    mlir::Value zero = constant(0, width);
    return builder.create<mlir::arith::SubIOp>(getLoc(), zero, operand);
  } else if (op == "!") {
    mlir::Value zero = constant(0, width);
    return builder.create<mlir::arith::CmpIOp>(getLoc(), mlir::arith::CmpIPredicate::eq, operand, zero);
  }
  toCout("MLIR gen: unsupported one_op: " + op);
  return mlir::Value();
}

mlir::Value MLIRUpdateFunctionGen::reduce_one_op_constraint(
    astNode* const node, uint32_t timeIdx, const uint32_t bound) {

  assert(node->childVec.size() == 1);
  mlir::Value operand = add_constraint(node->childVec[0], timeIdx, bound);
  if (!operand) return mlir::Value();

  uint32_t width = getWidth(operand);
  std::string op = node->op;

  if (op == "|" || op == "reduce_or") {
    mlir::Value zero = constant(0, width);
    return builder.create<mlir::arith::CmpIOp>(getLoc(), mlir::arith::CmpIPredicate::ne, operand, zero);
  } else if (op == "&" || op == "reduce_and") {
    mlir::Value allOnes = constant(width < 64 ? (1ULL << width) - 1 : ~0ULL, width);
    return builder.create<mlir::arith::CmpIOp>(getLoc(), mlir::arith::CmpIPredicate::eq, operand, allOnes);
  } else if (op == "^" || op == "reduce_xor") {
    mlir::Value result = extract_bits(operand, 0, 0);
    for (uint32_t i = 1; i < width; i++) {
      mlir::Value bit = extract_bits(operand, i, i);
      result = builder.create<mlir::arith::XOrIOp>(getLoc(), result, bit);
    }
    return result;
  } else if (op == "!" || op == "reduce_not" || op == "~|") {
    // Logical NOT / NOR reduction: result is 1 iff operand is 0
    mlir::Value zero = constant(0, width);
    return builder.create<mlir::arith::CmpIOp>(getLoc(), mlir::arith::CmpIPredicate::eq, operand, zero);
  }
  toCout("MLIR gen: unsupported reduce op: " + op);
  return mlir::Value();
}

mlir::Value MLIRUpdateFunctionGen::ite_op_constraint(
    astNode* const node, uint32_t timeIdx, const uint32_t bound) {

  toCoutVerb("MLIR ite_op for: " + node->dest);
  auto curMod = insContextStk.get_curMod();
  assert(node->srcVec.size() == 3 && node->childVec.size() == 3);

  std::string condAndSlice = node->srcVec[0];
  std::string condVar, condSlice;
  split_slice(condAndSlice, condVar, condSlice);

  mlir::Value cond;
  if (is_number(condVar)) {
    uint32_t w = get_var_slice_width_simp(condAndSlice, curMod);
    cond = literal_to_constant(condAndSlice, w);
  } else {
    mlir::Value tmp = add_constraint(node->childVec[0], timeIdx, bound);
    if (condSlice.empty() || has_direct_assignment(condAndSlice, curMod))
      cond = tmp;
    else {
      uint32_t hi = get_lgc_hi(condAndSlice, curMod);
      uint32_t lo = get_lgc_lo(condAndSlice, curMod);
      cond = extract_bits(tmp, hi, lo);
    }
  }

  mlir::Value trueVal = add_constraint(node->childVec[1], timeIdx, bound);
  mlir::Value falseVal = add_constraint(node->childVec[2], timeIdx, bound);

  if (!cond || !trueVal || !falseVal) return mlir::Value();

  if (getWidth(cond) != 1) {
    mlir::Value zero = constant(0, getWidth(cond));
    cond = builder.create<mlir::arith::CmpIOp>(getLoc(), mlir::arith::CmpIPredicate::ne, cond, zero);
  }

  uint32_t tw = getWidth(trueVal), fw = getWidth(falseVal);
  if (tw != fw) {
    uint32_t maxW = std::max(tw, fw);
    if (tw < maxW) trueVal = zext(trueVal, maxW);
    if (fw < maxW) falseVal = zext(falseVal, maxW);
  }

  return builder.create<mlir::SelectOp>(getLoc(), cond, trueVal, falseVal);
}

mlir::Value MLIRUpdateFunctionGen::sel_op_constraint(
    astNode* const node, uint32_t timeIdx, const uint32_t bound) {

  toCoutVerb("MLIR sel_op for: " + node->dest);
  auto curMod = insContextStk.get_curMod();
  assert(node->srcVec.size() == 3);

  if (node->op == "sel5") {
    uint32_t hi = std::stoi(node->srcVec[1]);
    uint32_t lo = std::stoi(node->srcVec[2]);
    mlir::Value opExpr = add_constraint(node->childVec[0], timeIdx, bound);
    if (!opExpr) return mlir::Value();
    return extract_bits(opExpr, hi, lo);
  }

  std::string op1AndSlice = node->srcVec[0];
  std::string op2AndSlice = node->srcVec[1];
  std::string integer = node->srcVec[2];
  std::string op2, op2Slice;
  split_slice(op2AndSlice, op2, op2Slice);

  mlir::Value op1Expr = add_constraint(node->childVec[0], timeIdx, bound);
  if (!op1Expr) return mlir::Value();

  if (is_number(op2)) {
    uint32_t startIdx = std::stoi(op2AndSlice);
    uint32_t len = std::stoi(integer);
    return extract_bits(op1Expr, startIdx + len - 1, startIdx);
  }

  mlir::Value op2Expr = add_constraint(node->childVec[1], timeIdx, bound);
  if (!op2Expr) return mlir::Value();
  uint32_t len = std::stoi(integer);
  uint32_t op1Width = getWidth(op1Expr);
  if (getWidth(op2Expr) != op1Width) op2Expr = zext(op2Expr, op1Width);
  mlir::Value shifted = builder.create<mlir::arith::ShRUIOp>(getLoc(), op1Expr, op2Expr);
  return trunc(shifted, len);
}

mlir::Value MLIRUpdateFunctionGen::src_concat_op_constraint(
    astNode* const node, uint32_t timeIdx, const uint32_t bound) {

  toCoutVerb("MLIR concat for: " + node->dest);
  auto curMod = insContextStk.get_curMod();
  if (node->childVec.empty()) return mlir::Value();

  mlir::Value result;
  for (uint32_t i = 0; i < node->childVec.size(); i++) {
    std::string srcAndSlice = node->srcVec[i];
    std::string src, srcSlice;
    split_slice(srcAndSlice, src, srcSlice);

    mlir::Value partExpr;
    if (is_number(src)) {
      uint32_t w = get_var_slice_width_simp(srcAndSlice, curMod);
      partExpr = literal_to_constant(srcAndSlice, w);
    } else {
      mlir::Value tmp = add_constraint(node->childVec[i], timeIdx, bound);
      if (srcSlice.empty() || has_direct_assignment(srcAndSlice, curMod))
        partExpr = tmp;
      else {
        uint32_t hi = get_lgc_hi(srcAndSlice, curMod);
        uint32_t lo = get_lgc_lo(srcAndSlice, curMod);
        partExpr = extract_bits(tmp, hi, lo);
      }
    }
    if (!partExpr) continue;
    if (!result) result = partExpr;
    else result = concat_values(result, partExpr);
  }
  return result;
}

mlir::Value MLIRUpdateFunctionGen::num_constraint(
    astNode* const node, uint32_t timeIdx) {
  auto curMod = insContextStk.get_curMod();
  std::string numStr = node->dest;
  uint32_t width = get_var_slice_width_simp(numStr, curMod);
  return literal_to_constant(numStr, width);
}

mlir::Value MLIRUpdateFunctionGen::input_constraint(
    astNode* const node, uint32_t timeIdx, const uint32_t bound) {

  auto curMod = insContextStk.get_curMod();
  std::string varAndSlice = node->dest;
  std::string var, varSlice;
  split_slice(varAndSlice, var, varSlice);

  std::string name = var + post_fix(timeIdx);
  mlir::Value v = lookupValue(name);
  if (!v) { name = var + post_fix(bound); v = lookupValue(name); }

  if (v) {
    if (!varSlice.empty()) {
      uint32_t hi = get_end(varSlice);
      uint32_t lo = get_begin(varSlice);
      return extract_bits(v, hi, lo);
    }
    return v;
  }
  toCout("MLIR gen: input not found: " + varAndSlice + " at t=" + std::to_string(timeIdx));
  return mlir::Value();
}

mlir::Value MLIRUpdateFunctionGen::add_nb_constraint(
    astNode* const node, uint32_t timeIdx, const uint32_t bound) {

  auto curMod = insContextStk.get_curMod();
  auto curDynData = get_dyn_data(curMod);
  std::string destAndSlice = node->dest;
  std::string dest, destSlice;
  split_slice(destAndSlice, dest, destSlice);

  if (timeIdx < bound) {
    toCoutVerb("MLIR nb_constraint for: " + destAndSlice + " t:" + std::to_string(timeIdx));
    auto srcExpr = add_constraint(node->childVec.front(), timeIdx + 1, bound);
    std::string srcAndSlice = node->srcVec[0];
    std::string src, srcSlice;
    split_slice(srcAndSlice, src, srcSlice);
    if (srcSlice.empty() || has_direct_assignment(srcAndSlice, curMod))
      return srcExpr;
    else {
      uint32_t srcHi = get_lgc_hi(srcAndSlice, curMod);
      uint32_t srcLo = get_lgc_lo(srcAndSlice, curMod);
      return extract_bits(srcExpr, srcHi, srcLo);
    }
  }

  // At boundary: return function argument
  if (g_use_read_ASV && is_read_asv(dest, curMod)) {
    if (!curDynData->isFunctionedSubMod) {
      std::string prefix = insContextStk.get_hier_name(false);
      if (!prefix.empty()) prefix += ".";
      dest = prefix + dest;
    }
    std::string destTimed = timed_name(dest, timeIdx);
    mlir::Value destExpr = lookupValue(destTimed);
    if (!destExpr) { toCout("MLIR: read ASV not found: " + destTimed); return mlir::Value(); }
    if (destSlice.empty()) return destExpr;
    uint32_t hi = get_end(destSlice);
    uint32_t lo = get_begin(destSlice);
    return extract_bits(destExpr, hi, lo);
  } else if (g_use_read_ASV) {
    uint32_t width = insContextStk.get_var_slice_width_simp(destAndSlice);
    std::string rstVal = get_rst_value(destAndSlice, timeIdx, width);
    return literal_to_constant(rstVal, width);
  } else {
    std::string destTimed = timed_name(dest, timeIdx);
    mlir::Value v = lookupValue(destTimed);
    if (v) return v;
    toCout("MLIR: register not found: " + destTimed);
    return mlir::Value();
  }
}

// ============================================================================
// case_constraint: chain of selects based on one-hot case variable bits
// Mirrors UpdateFunctionGen::case_constraint and add_one_case_branch_expr
// ============================================================================

mlir::Value MLIRUpdateFunctionGen::case_constraint(
    astNode* const node, uint32_t timeIdx, const uint32_t bound) {

  toCoutVerb("MLIR case_constraint for: " + node->dest);
  auto curMod = insContextStk.get_curMod();
  assert(node->type == CASE);
  assert(node->srcVec.size() % 2 == 1); // caseVar, {caseVal, assignVar}*, defaultAssignVar

  std::string destAndSlice = node->dest;
  uint32_t destWidthNum = get_var_slice_width_simp(destAndSlice, curMod);

  // Get the case variable expression
  std::string caseVarAndSlice = node->srcVec[0];
  std::string caseVar, caseVarSlice;
  split_slice(caseVarAndSlice, caseVar, caseVarSlice);
  uint32_t caseHi = get_lgc_hi(caseVarAndSlice, curMod);
  uint32_t caseLo = get_lgc_lo(caseVarAndSlice, curMod);

  mlir::Value caseVarExpr;
  if (caseVarSlice.empty() || has_direct_assignment(caseVarAndSlice, curMod))
    caseVarExpr = add_constraint(node->childVec[0], timeIdx, bound);
  else {
    mlir::Value tmp = add_constraint(node->childVec[0], timeIdx, bound);
    caseVarExpr = extract_bits(tmp, caseHi, caseLo);
  }

  if (!caseVarExpr) return constant(0, destWidthNum);

  // Build chain of selects: process case branches from first to default
  // srcVec layout: [caseVar, caseVal0, assignVar0, caseVal1, assignVar1, ..., defaultAssignVar]
  // childVec layout: [caseVarChild, thenChild, defaultChild]

  // Helper lambda to build a condition from a case value
  auto makeCaseCond = [&](const std::string &caseVal, mlir::Value caseExpr) -> mlir::Value {
    // If value has '?' (don't-care bits with one-hot encoding), use get_pos_of_one
    if (caseVal.find("?") != std::string::npos && caseVal.find("b") != std::string::npos) {
      uint32_t posOfOne = get_pos_of_one(caseVal);
      mlir::Value caseBit = extract_bits(caseExpr, posOfOne, posOfOne);
      mlir::Value oneVal = constant(1, 1);
      return builder.create<mlir::arith::CmpIOp>(getLoc(),
          mlir::arith::CmpIPredicate::eq, caseBit, oneVal);
    }
    // Otherwise do a full equality comparison
    uint32_t caseWidth = getWidth(caseExpr);
    mlir::Value label = literal_to_constant(caseVal, caseWidth);
    return builder.create<mlir::arith::CmpIOp>(getLoc(),
        mlir::arith::CmpIPredicate::eq, caseExpr, label);
  };

  // Get the "then" value for first case
  std::string firstCaseVal = node->srcVec[1];
  mlir::Value iteCond = makeCaseCond(firstCaseVal, caseVarExpr);

  // First branch value
  std::string assignVarAndSlice = node->srcVec[2];
  uint32_t hi = get_lgc_hi(assignVarAndSlice, curMod);
  uint32_t lo = get_lgc_lo(assignVarAndSlice, curMod);
  std::string assignVar, assignSlice;
  split_slice(assignVarAndSlice, assignVar, assignSlice);

  mlir::Value thenRet;
  if (isNum(assignVarAndSlice)) {
    uint32_t w = get_var_slice_width_simp(assignVarAndSlice, curMod);
    thenRet = literal_to_constant(assignVarAndSlice, w);
  } else {
    mlir::Value tmp = add_constraint(node->childVec[1], timeIdx, bound);
    if (tmp) {
      uint32_t tmpW = getWidth(tmp);
      if (hi >= tmpW && lo == 0)
        thenRet = tmp;
      else
        thenRet = extract_bits(tmp, hi, lo);
    } else {
      thenRet = constant(0, destWidthNum);
    }
  }

  // Build the else chain recursively (via iteration)
  // Process remaining branches: srcVec[3], srcVec[4], srcVec[5], srcVec[6], ...
  mlir::Value elseRet;
  uint32_t idx = 3;
  if (idx < node->srcVec.size() - 1) {
    // More case branches
    // For simplicity, build from last to first
    // First get the default value (last element in srcVec)
    std::string defaultAssign = node->srcVec.back();

    if (defaultAssign == "default") {
      // The default branch uses childVec[2]
      mlir::Value tmp = add_constraint(node->childVec[2], timeIdx, bound);
      elseRet = tmp ? tmp : constant(0, destWidthNum);
    } else if (isNum(defaultAssign)) {
      uint32_t w = get_var_slice_width_simp(defaultAssign, curMod);
      elseRet = literal_to_constant(defaultAssign, w);
    } else {
      uint32_t dhi = get_lgc_hi(defaultAssign, curMod);
      uint32_t dlo = get_lgc_lo(defaultAssign, curMod);
      mlir::Value tmp = add_constraint(node->childVec[2], timeIdx, bound);
      if (tmp) {
        uint32_t tmpW = getWidth(tmp);
        if (dhi >= tmpW && dlo == 0) elseRet = tmp;
        else elseRet = extract_bits(tmp, dhi, dlo);
      } else {
        elseRet = constant(0, destWidthNum);
      }
    }

    // Process intermediate branches backwards (skip last element which is default)
    int lastCaseIdx = static_cast<int>(node->srcVec.size()) - 2;
    // The last assignVar is the default, already handled above.
    // Intermediate branches are at indices 3,4  5,6  ... lastCaseIdx-1,lastCaseIdx-0
    for (int i = lastCaseIdx - 1; i >= 3; i -= 2) {
      std::string branchCaseVal = node->srcVec[i];
      std::string branchAssign = node->srcVec[i + 1];
      if (branchAssign == "default") continue;
      mlir::Value brCond = makeCaseCond(branchCaseVal, caseVarExpr);

      uint32_t bhi = get_lgc_hi(branchAssign, curMod);
      uint32_t blo = get_lgc_lo(branchAssign, curMod);

      mlir::Value branchVal;
      if (isNum(branchAssign)) {
        uint32_t w = get_var_slice_width_simp(branchAssign, curMod);
        branchVal = literal_to_constant(branchAssign, w);
      } else {
        mlir::Value tmp = add_constraint(node->childVec[1], timeIdx, bound);
        if (tmp) {
          uint32_t tmpW = getWidth(tmp);
          if (bhi >= tmpW && blo == 0) branchVal = tmp;
          else branchVal = extract_bits(tmp, bhi, blo);
        } else {
          branchVal = constant(0, destWidthNum);
        }
      }

      // Width-match
      if (branchVal && elseRet) {
        uint32_t bw = getWidth(branchVal), ew = getWidth(elseRet);
        if (bw != ew) {
          uint32_t maxW = std::max(bw, ew);
          if (bw < maxW) branchVal = zext(branchVal, maxW);
          if (ew < maxW) elseRet = zext(elseRet, maxW);
        }
      }

      elseRet = builder.create<mlir::SelectOp>(getLoc(), brCond, branchVal, elseRet);
    }
  } else {
    // Only default branch
    std::string defaultAssign = node->srcVec.back();

    if (defaultAssign == "default") {
      mlir::Value tmp = add_constraint(node->childVec[2], timeIdx, bound);
      elseRet = tmp ? tmp : constant(0, destWidthNum);
    } else if (isNum(defaultAssign)) {
      uint32_t w = get_var_slice_width_simp(defaultAssign, curMod);
      elseRet = literal_to_constant(defaultAssign, w);
    } else {
      uint32_t dhi = get_lgc_hi(defaultAssign, curMod);
      uint32_t dlo = get_lgc_lo(defaultAssign, curMod);
      mlir::Value tmp = add_constraint(node->childVec[2], timeIdx, bound);
      if (tmp) {
        uint32_t tmpW = getWidth(tmp);
        if (dhi >= tmpW && dlo == 0) elseRet = tmp;
        else elseRet = extract_bits(tmp, dhi, dlo);
      } else {
        elseRet = constant(0, destWidthNum);
      }
    }
  }

  if (!thenRet || !elseRet) return constant(0, destWidthNum);

  // Width-match then and else
  uint32_t tw = getWidth(thenRet), ew = getWidth(elseRet);
  if (tw != ew) {
    uint32_t maxW = std::max(tw, ew);
    if (tw < maxW) thenRet = zext(thenRet, maxW);
    if (ew < maxW) elseRet = zext(elseRet, maxW);
  }

  return builder.create<mlir::SelectOp>(getLoc(), iteCond, thenRet, elseRet);
}

// ============================================================================
// switch_constraint: lookup table using memref.global + memref.load
// The switch maps consecutive index values to output values.
// In MLIR, we represent this as a global memref and index-based load.
// ============================================================================

mlir::Value MLIRUpdateFunctionGen::switch_constraint(
    astNode* const node, uint32_t timeIdx, const uint32_t bound) {

  toCoutVerb("MLIR switch_constraint for: " + node->dest);
  auto curMod = insContextStk.get_curMod();

  std::string switchVarAndSlice = node->srcVec[0];
  std::string switchVar, switchVarSlice;
  split_slice(switchVarAndSlice, switchVar, switchVarSlice);
  uint32_t hi = get_lgc_hi(switchVarAndSlice, curMod);
  uint32_t lo = get_lgc_lo(switchVarAndSlice, curMod);

  // Get the switch index expression (at timeIdx+1 like the LLVM path)
  mlir::Value switchVarExpr;
  if (switchVarSlice.empty() || has_direct_assignment(switchVarAndSlice, curMod))
    switchVarExpr = add_constraint(node->childVec[0], timeIdx + 1, bound);
  else {
    mlir::Value tmp = add_constraint(node->childVec[0], timeIdx + 1, bound);
    switchVarExpr = extract_bits(tmp, hi, lo);
  }

  if (!switchVarExpr) {
    uint32_t destW = get_var_slice_width_simp(node->dest, curMod);
    return constant(0, destW);
  }

  // Get the switch table data
  std::string destAndSlice = node->dest;
  auto switchInfo = curMod->switchTable[destAndSlice];
  auto &assignVec = switchInfo.assignVec;
  uint32_t size = assignVec.size();

  std::string firstAssignVal = assignVec.front().second;
  uint32_t assignValueWidth = get_formed_width(firstAssignVal);

  // Build a chain of selects to implement the lookup table
  // This is cleaner for MLIR than a global array + GEP
  mlir::Value result = literal_to_constant(assignVec.back().second, assignValueWidth);

  for (int i = static_cast<int>(size) - 1; i >= 0; i--) {
    uint32_t switchValue = hdb2int(assignVec[i].first);
    uint32_t assignValue = hdb2int(assignVec[i].second);

    mlir::Value caseVal = constant(switchValue, getWidth(switchVarExpr));
    mlir::Value cond = builder.create<mlir::arith::CmpIOp>(getLoc(),
        mlir::arith::CmpIPredicate::eq, switchVarExpr, caseVal);
    mlir::Value branchVal = constant(assignValue, assignValueWidth);

    result = builder.create<mlir::SelectOp>(getLoc(), cond, branchVal, result);
  }

  return result;
}

// ============================================================================
// dyn_sel_constraint: dynamic memory read → memref.load
// Represents array[index] access using MLIR memref operations.
// ============================================================================

mlir::Value MLIRUpdateFunctionGen::dyn_sel_constraint(
    astNode* const node, uint32_t timeIdx, const uint32_t bound) {

  toCoutVerb("MLIR dyn_sel_constraint for: " + node->dest);
  auto curMod = insContextStk.get_curMod();

  std::string destAndSlice = node->dest;
  std::string mem = node->srcVec[0];
  std::string addrAndSlice = node->srcVec[1];
  uint32_t addrHi = get_lgc_hi(addrAndSlice, curMod);
  uint32_t addrLo = get_lgc_lo(addrAndSlice, curMod);

  std::string addr, addrSlice;
  split_slice(addrAndSlice, addr, addrSlice);

  // Get address expression
  mlir::Value tmpExpr = add_constraint(node->childVec[1], timeIdx, bound);
  mlir::Value addrExpr;
  if (addrSlice.empty() || has_direct_assignment(addrAndSlice, curMod))
    addrExpr = tmpExpr;
  else
    addrExpr = extract_bits(tmpExpr, addrHi, addrLo);

  if (!addrExpr) {
    uint32_t destW = get_var_slice_width_simp(destAndSlice, curMod);
    return constant(0, destW);
  }

  if (curMod->moduleMems.find(mem) == curMod->moduleMems.end()) {
    toCout("MLIR error: not memory for dyn_sel: " + mem);
    abort();
  }

  uint32_t lineWidth = curMod->moduleMems[mem].first;
  uint32_t lineNum = curMod->moduleMems[mem].second;

  // Create or find the memref for this memory buffer
  // In MLIR, we allocate a local memref and load from it
  std::string memKey = mem + "_mlir_buf";
  mlir::Value memBuf = lookupValue(memKey);

  if (!memBuf) {
    // Allocate a local memref for this buffer
    auto elemTy = builder.getIntegerType(lineWidth);
    auto memrefTy = mlir::MemRefType::get({static_cast<int64_t>(lineNum)}, elemTy);

    // Save current insertion point, create alloc at function entry
    auto savedIP = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(entryBlock);
    memBuf = builder.create<mlir::memref::AllocaOp>(getLoc(), memrefTy);
    builder.restoreInsertionPoint(savedIP);

    registerValue(memKey, memBuf);
  }

  // Process any pending memory writes
  auto childNode = node->childVec[0];
  if (memNodeMap.find(mem) == memNodeMap.end())
    memNodeMap.emplace(mem, childNode);

  mem_assign_constraint(childNode, timeIdx, bound);

  // Convert integer address to index type for memref.load
  mlir::Value idxVal = builder.create<mlir::arith::IndexCastOp>(getLoc(),
      builder.getIndexType(), addrExpr);

  return builder.create<mlir::memref::LoadOp>(getLoc(), memBuf,
      mlir::ValueRange{idxVal});
}

// ============================================================================
// mem_assign_constraint: conditional memory write → scf.if + memref.store
// ============================================================================

void MLIRUpdateFunctionGen::mem_assign_constraint(
    astNode* const node, uint32_t timeIdx, const uint32_t bound) {

  toCoutVerb("MLIR mem_assign_constraint for: " + node->dest);
  // TODO: implement full memory write semantics with scf.if
  // For now this is a no-op — the memory contents are zero-initialized
  // and reads return from the alloca'd buffer.
}

// ============================================================================
// submod_constraint: sub-module handling
// For non-functionalized sub-modules, push context and continue.
// For functionalized ones, create/call a sub-function.
// ============================================================================

mlir::Value MLIRUpdateFunctionGen::submod_constraint(
    astNode* const node, uint32_t timeIdx, const uint32_t bound) {

  toCoutVerb("MLIR submod_constraint for: " + node->dest);
  auto curMod = insContextStk.get_curMod();
  std::string destAndSlice = node->dest;
  std::string curTarget = insContextStk.get_target();
  auto curDynData = get_dyn_data(curMod);
  auto pair = curMod->wire2InsPortMp[destAndSlice];
  std::string insName = pair.first;
  std::string outPort = pair.second;

  auto subMod = get_mod_info(insName, curMod);
  auto subDynData = get_dyn_data(subMod);

  // If the submod is not functionalized, push context and recurse
  if (!subDynData->isFunctionedSubMod) {
    Context_t insCntxt(insName, curTarget, subMod, curMod, nullptr);
    insContextStk.push_back(insCntxt);
    auto expr = add_constraint(outPort, timeIdx, bound);
    insContextStk.pop_back();
    return expr;
  }

  // For functionalized sub-modules with memory, return the function argument
  std::string modName = subMod->name;
  if (is_mem_module(modName)) {
    std::string portName = insName + "." + outPort + post_fix(bound);
    mlir::Value v = lookupValue(portName);
    if (v) return v;
    toCout("MLIR: mem module port not found: " + portName);
    uint32_t destW = get_var_slice_width_simp(destAndSlice, curMod);
    return constant(0, destW);
  }

  // For other functionalized sub-modules, inline their logic
  // by pushing context and recursing (simpler than creating sub-functions)
  initialize_min_delay(subMod, outPort);

  Context_t insCntxt(insName, curTarget, subMod, curMod, nullptr);
  subDynData->isFunctionedSubMod = false; // treat as inlined
  insContextStk.push_back(insCntxt);
  auto expr = add_constraint(outPort, timeIdx, bound);
  insContextStk.pop_back();
  subDynData->isFunctionedSubMod = true; // restore

  return expr;
}

// ============================================================================
// Helper functions
// ============================================================================

std::shared_ptr<ModuleDynInfo_t>
MLIRUpdateFunctionGen::get_dyn_data(std::shared_ptr<ModuleInfo_t> curMod) {
  auto it = dynDataMp.find(curMod->name);
  if (it != dynDataMp.end()) return it->second;
  auto data = std::make_shared<ModuleDynInfo_t>();
  dynDataMp[curMod->name] = data;
  return data;
}

void MLIRUpdateFunctionGen::clean_all_mod_dynamic_info() {
  for (auto &p : dynDataMp) p.second->clean_ir_data();
}

void MLIRUpdateFunctionGen::clean_data() {
  CLEAN_QUEUE.clear(); DIRTY_QUEUE.clear(); resetedReg.clear();
}

void MLIRUpdateFunctionGen::push_clean_queue(astNode* node, uint32_t t) { CLEAN_QUEUE[node] = t; }
void MLIRUpdateFunctionGen::push_dirty_queue(astNode* node, uint32_t t) { DIRTY_QUEUE[node] = t; }
bool MLIRUpdateFunctionGen::is_in_clean_queue(std::string var) { return false; }
bool MLIRUpdateFunctionGen::is_in_dirty_queue(std::string var) { return false; }

void MLIRUpdateFunctionGen::initialize_min_delay(
    std::shared_ptr<ModuleInfo_t> &modInfo, std::string outPort) {
  auto dynData = get_dyn_data(modInfo);
}

} // namespace funcExtract
