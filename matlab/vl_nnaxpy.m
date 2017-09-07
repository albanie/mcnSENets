function y = vl_nnaxpy(x,y,z,varargin)
%VL_NNAXPY CNN 
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]
  [~, dzdy] = vl_argparsepos(struct(), varargin) ;

  if isempty(dzdy)
    y = bsxfun(@times, x, y) + z ;
  else
    keyboard
    base = 1 / (size(x,1) * size(x,2)) * ones(size(x), 'like', x) ;
    y = bsxfun(@times, base, dzdy) ;
  end
