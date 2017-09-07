classdef Axpy < dagnn.Filter

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnaxpy(inputs{1}, inputs{2}, inputs{3}) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnaxpy(inputs{1}, inputs{2}, inputs{3}, derOutputs{1}) ;
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = inputSizes{1}(3) ;
    end

    function obj = Pooling(varargin)
      obj.load(varargin) ;
    end
  end
end
