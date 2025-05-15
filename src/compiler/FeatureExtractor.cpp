#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <vector>
#include <string>
#include <map>
#include <fstream>

using namespace llvm;

namespace {

struct FeatureExtractor : public ModulePass {
  static char ID;
  FeatureExtractor() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    // Create a JSON object to store all extracted features
    std::map<std::string, std::map<std::string, double>> moduleFeatures;
    
    // For each function in the module
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;  // Skip function declarations without bodies
      
      // Extract features for this function
      std::map<std::string, double> features = extractFunctionFeatures(F);
      
      // Add to module features
      moduleFeatures[F.getName().str()] = features;
    }
    
    // Convert to JSON and write to file
    std::error_code EC;
    raw_fd_ostream OS("features.json", EC, sys::fs::OF_None);
    if (EC) {
      errs() << "Error opening features.json file: " << EC.message() << "\n";
      return false;
    }
    
    // Manual JSON construction since LLVM's JSON lib is basic
    OS << "{\n";
    bool firstFunc = true;
    for (auto &FuncFeatures : moduleFeatures) {
      if (!firstFunc) OS << ",\n";
      firstFunc = false;
      
      OS << "  \"" << FuncFeatures.first << "\": {\n";
      bool firstFeature = true;
      for (auto &Feature : FuncFeatures.second) {
        if (!firstFeature) OS << ",\n";
        firstFeature = false;
        
        OS << "    \"" << Feature.first << "\": " << Feature.second;
      }
      OS << "\n  }";
    }
    OS << "\n}\n";
    
    return false;  // We're not modifying the module
  }

private:
  std::map<std::string, double> extractFunctionFeatures(Function &F) {
    std::map<std::string, double> features;
    
    // Basic metrics
    features["basic_block_count"] = F.size();  // Number of basic blocks
    
    // Instruction metrics
    unsigned totalInsts = 0;
    unsigned branchCount = 0;
    unsigned loadCount = 0;
    unsigned storeCount = 0;
    unsigned arithmeticCount = 0;
    unsigned callCount = 0;
    unsigned phiCount = 0;
    
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      totalInsts++;
      
      if (isa<BranchInst>(&*I) || isa<SwitchInst>(&*I))
        branchCount++;
      else if (isa<LoadInst>(&*I))
        loadCount++;
      else if (isa<StoreInst>(&*I))
        storeCount++;
      else if (isa<BinaryOperator>(&*I))
        arithmeticCount++;
      else if (isa<CallInst>(&*I) || isa<InvokeInst>(&*I))
        callCount++;
      else if (isa<PHINode>(&*I))
        phiCount++;
    }
    
    features["instruction_count"] = totalInsts;
    features["branch_count"] = branchCount;
    features["load_count"] = loadCount;
    features["store_count"] = storeCount;
    features["arithmetic_count"] = arithmeticCount;
    features["call_count"] = callCount;
    features["phi_count"] = phiCount;
    
    // Derived metrics
    features["branch_density"] = totalInsts > 0 ? (double)branchCount / totalInsts : 0;
    features["memory_density"] = totalInsts > 0 ? (double)(loadCount + storeCount) / totalInsts : 0;
    features["arithmetic_density"] = totalInsts > 0 ? (double)arithmeticCount / totalInsts : 0;
    features["avg_insts_per_bb"] = F.size() > 0 ? (double)totalInsts / F.size() : 0;
    
    // Loop analysis could be added here
    
    return features;
  }
};

char FeatureExtractor::ID = 0;

// Register the pass
static RegisterPass<FeatureExtractor> X("extract-features", 
                                        "Extract code features for ML-based optimization");

// Register the pass in the pass manager
static void registerFeatureExtractorPass(const PassManagerBuilder &,
                                        legacy::PassManagerBase &PM) {
  PM.add(new FeatureExtractor());
}

static RegisterStandardPasses RegisterFeatureExtractor(
    PassManagerBuilder::EP_OptimizerLast,
    registerFeatureExtractorPass);

}  // namespace
