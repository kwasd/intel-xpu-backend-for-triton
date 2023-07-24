#ifndef TRITON_TRITONSPIRVTOLLVM_H
#define TRITON_TRITONSPIRVTOLLVM_H

//#include "TritonGPUToSPIRVBase.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
// using namespace mlir::triton;

void populateSPIRVToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 mlir::MLIRContext *context,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit);

#endif
