#include "DotOpToSPIRV.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

LogicalResult convertFMASPIRVDot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            TritonGPUToSPIRVTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter);

struct DotOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::DotOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::DotOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    if (D.getType()
            .cast<RankedTensorType>()
            .getEncoding()
            .isa<BlockedEncodingAttr>())
      return convertFMASPIRVDot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to SPIRV.");
  }
};

void populateDotOpToSPIRVPatterns(TritonGPUToSPIRVTypeConverter &typeConverter,
                                  mlir::MLIRContext *context,
                                  RewritePatternSet &patterns, int numWarps,
                                  ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                  ModuleAllocation &allocation,
                                  PatternBenefit benefit) {
  patterns.add<DotOpSPIRVConversion>(typeConverter, context, allocation,
                                     benefit);
}
