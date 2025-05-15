#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Process.h"
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <memory>
#include <array>

using namespace llvm;

namespace {

// Helper function to execute Python ML model and get optimization decision
int getOptimizationDecision(const std::string &functionName) {
  // The actual implementation would call the Python ML model
  // For now, we'll use a simple system call to the Python script
  
  std::string command = "python3 " + 
                       (sys::Process::GetEnv("COMPILEML_PATH") ? 
                        *sys::Process::GetEnv("COMPILEML_PATH") : ".") + 
                       "/src/ml/predict_optimization.py " + functionName;
  
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
  
  if (!pipe) {
    errs() << "Failed to run ML prediction script\n";
    return 3; // Default to -O3 on failure
  }
  
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  
  // Parse the result (should be an integer 0-3 for O0-O3)
  try {
    return std::stoi(result);
  } catch (...) {
    errs() << "Failed to parse ML prediction result\n";
    return 3; // Default to -O3 on failure
  }
}

struct OptimizationPass : public ModulePass {
  static char ID;
  OptimizationPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    bool modified = false;
    
    // For each function in the module
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;  // Skip function declarations without bodies
      
      // Get the optimization level from ML model
      int optLevel = getOptimizationDecision(F.getName().str());
      
      // Store the recommended optimization level as metadata
      LLVMContext &Ctx = F.getContext();
      MDNode *MD = MDNode::get(Ctx, 
                          ConstantAsMetadata::get(ConstantInt::get(
                            Type::getInt32Ty(Ctx), optLevel)));
      
      F.setMetadata("ml.optlevel", MD);
      
      errs() << "ML: Function " << F.getName() << " -> O" << optLevel << "\n";
      modified = true;
    }
    
    return modified;
  }
};

char OptimizationPass::ID = 0;

// Register the pass
static RegisterPass<OptimizationPass> X("ml-optimize", 
                                       "Apply ML-based optimization selection");

// Register the pass in the pass manager
static void registerOptimizationPass(const PassManagerBuilder &,
                                    legacy::PassManagerBase &PM) {
  PM.add(new OptimizationPass());
}

static RegisterStandardPasses RegisterOptimizationPass(
    PassManagerBuilder::EP_ModuleOptimizerEarly,
    registerOptimizationPass);

}  // namespace
