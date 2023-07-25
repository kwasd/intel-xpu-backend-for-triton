# SPIRV Tools is required to disassemble the spirv module
find_package(SPIRV_Tools)

# SPIRV-LLVM Translator is required.
find_package(SPIRVToLLVMTranslator)

add_mlir_translation_library(TritonSPIRV
        SPIRVTranslation.cpp

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRLLVMIRTransforms
        MLIRSupport
        MLIRTargetLLVMIRExport
        # spirv tools
        SPIRV-Tools
        SPIRV-Tools-link
        LLVMSPIRVLib
        )

# Add SPIRVLLVMTranslator include dir
target_include_directories(TritonSPIRV PRIVATE ${SPIRVToLLVMTranslator_INCLUDE_DIR})
