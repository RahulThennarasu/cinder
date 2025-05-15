add_library(FeatureExtractor MODULE
  FeatureExtractor.cpp
)

add_library(OptimizationPass MODULE
  OptimizationPass.cpp
)

# Get proper shared library suffix
if(APPLE)
  set(SHARED_LIB_SUFFIX ".dylib")
else()
  set(SHARED_LIB_SUFFIX ".so")
endif()

# Set output names
set_target_properties(FeatureExtractor PROPERTIES
  PREFIX ""
  OUTPUT_NAME "FeatureExtractor"
  SUFFIX "${SHARED_LIB_SUFFIX}"
)

set_target_properties(OptimizationPass PROPERTIES
  PREFIX ""
  OUTPUT_NAME "OptimizationPass"
  SUFFIX "${SHARED_LIB_SUFFIX}"
)

# Link against LLVM libraries
target_link_libraries(FeatureExtractor
  ${LLVM_AVAILABLE_LIBS}
)

target_link_libraries(OptimizationPass
  ${LLVM_AVAILABLE_LIBS}
)

# Installation targets
install(TARGETS FeatureExtractor OptimizationPass
  LIBRARY DESTINATION lib
)
