# SPIRV-LLVM Translator is required.
find_package(SPIRVToLLVMTranslator)

add_mlir_translation_library(TritonSPIRV
        SPIRVTranslation.cpp

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        TritonLLVMIR
        # spirv tools
        LLVMSPIRVLib
        )

# Add SPIRVLLVMTranslator include dir
target_include_directories(TritonSPIRV PRIVATE ${SPIRVToLLVMTranslator_INCLUDE_DIR})
