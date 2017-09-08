function net = cnn_imagenet_init_senet(opts)
%CNN_IMAGENET_INIT_SENET Initialize the SE-networks for ImageNet classification
%   NET = CNN_IMAGENET_INIT_SENET initialises a Squeeze-and-Excitation Network
%   for training on imagenet.

%   This script is based on the cnn_imagenet_init_resnet.m script by 
%   Andrea Vedaldi.
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  data = Input('input') ; label = Input('label') ;

  % define a placeholder for depth (which can be filled in later during
  % network construction
  dd = -1 ; 

% -------------------------------------------------------------------------
% Add input section
% -------------------------------------------------------------------------

   c1 = add_block('conv1', data, opts, [7, 7, 3, 64], 1, 1) ;
   net = vl_nnpool(c1, [3 3], 'stride', 2, 'pad', 1, 'method', 'max') ;
   net.name = 'conv1_pool' ;

% -------------------------------------------------------------------------
% Add intermediate sections
% -------------------------------------------------------------------------

  for s = 2:5
    switch s
      case 2, sectionLen = 3 ;
      case 3, sectionLen = 4 ; % 8 ;
      case 4, sectionLen = 6 ; % 23 ; % 36 ;
      case 5, sectionLen = 3 ;
    end

    % -----------------------------------------------------------------------
    % Add intermediate segments for each section
    for l = 1:sectionLen
      name = sprintf('conv%d_%d', s, l)  ;
      skip = net ;
      if l == 1 % Optional adapter layer
        f = [1, 1, dd, 2^(s+6)] ; down = s >= 3 ; useRelu = 0 ; 
        skip = add_block([name '_adapt_conv'], skip, opts, f, useRelu, down) ;
      end

      down = (s >= 3) & l == 1 ;
      net = add_block([name 'a'], net, opts, [1,1,dd,2^(s+4)], 1, 0) ;
      net = add_block([name 'b'], net, opts, [3,3,dd,2^(s+4)], 1, down) ;
      net = add_block([name 'c'], net, opts, [1,1,dd,2^(s+6)], 0, 0) ;

      % relu and sum layers (TODO: switch to preactivations)
      net = Layer.create(@vl_nnsum, {skip, net}) ; net.name = [name '_sum'] ;
      net = vl_nnrelu(net) ; net.name = [name '_relu'] ; 
    end
  end

  net = vl_nnpool(net, [7 7], 'stride', 2, 'pad', 1, 'method', 'avg') ;
  net.name = 'prediction_avg' ;
  preds = add_conv(net, [1, 1, 2048, 1000], down, opts) ;
  preds.name = 'prediction' ;
  objective = Layer.create(@vl_nnloss, {preds, label}, 'numInputDer', 1) ;
  objective.name = 'objective' ; 
  largs = {preds, label, 'loss', 'classerror'} ; args = {'numInputDer', 0} ;
  top1 = Layer.create(@vl_nnloss, largs, args{:}) ; top1.name = 'top1error' ;
  largs = {preds, label, 'loss', 'topkerror', 'opts', {'topK', 5}} ; 
  top5 = Layer.create(@vl_nnloss, largs, args{:}) ; top5.name = 'top5error' ;

  % For uniformity with the other ImageNet networks,
  % the input data is *not* normalized to have unit standard deviation,
  % whereas this is enforced by batch normalization deeper down.
  % The ImageNet standard deviation (for each of R, G, and B) is about 60, so
  % we adjust the weights and learing rate accordingly in the first layer.
  %
  % This simple change improves performance almost +1% top 1 error.
  
  fSc = 1 / 100 ; lrSc = 1 / 100^2 ;
  f = objective.find('conv1_conv', 1).inputs{2} ;
  objective.find('conv1_conv', 1).inputs{2}.value = f.value * fSc ;
  objective.find('conv1_conv', 1).inputs{2}.learningRate = f.learningRate * lrSc ;

  % Taken from Andrea's code:
  bnorm_layers = objective.find(@vl_nnbnorm_wrapper) ;
  for ii = 1:numel(bnorm_layers)
    bnorm_layers{ii}.inputs{4}.learningRate = 0.3 ;
  end
  net = Net(objective, top1, top5) ; % compile

  net = toDagNN(net) ; % return to dag for multi-gpu efficiency

% -------------------------------------------------------------------------
%                                                           Meta parameters
% -------------------------------------------------------------------------

  net.meta.normalization.imageSize = [224 224 3] ;
  net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;
  net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 256 ;
  net.meta.normalization.averageImage = opts.modelOpts.averageImage ;

  net.meta.classes.name = opts.modelOpts.classNames ;
  net.meta.classes.description = opts.modelOpts.classDescriptions ;

  augmentation.jitterFlip = true ;
  augmentation.jitterLocation = true ;
  augmentation.jitterAspect = [3/4, 4/3] ;
  augmentation.jitterScale  = [0.4, 1.1] ;
  augmentation.jitterBrightness = double(0.1 * opts.modelOpts.colorDeviation) ;
  net.meta.augmentation = augmentation ;
  %net.meta.augmentation.jitterSaturation = 0.4 ;
  %net.meta.augmentation.jitterContrast = 0.4 ;

  net.meta.inputSize = {'input', [net.meta.normalization.imageSize 32]} ;

  %lr = logspace(-1, -3, 60) ;
  lr = [0.1 * ones(1,30), 0.01*ones(1,30), 0.001*ones(1,30)] ;
  trainOpts.learningRate = lr ;
  trainOpts.numEpochs = numel(lr) ;
  trainOpts.momentum = 0.9 ;
  trainOpts.batchSize = 256 ;
  trainOpts.numSubBatches = 4 ;
  trainOpts.weightDecay = 0.0001 ;
  net.meta.trainOpts = trainOpts ;


% --------------------------------------------------------------------
function net = add_block(name, net, opts, sz, useRelu, down, varargin)
% --------------------------------------------------------------------
  if sz(3) == -1 % handle dd - update to match previous conv
    sz(3) = size(net.find(@vl_nnconv, 1).inputs{2}.value, 4) ; 
  end
  net = add_conv(net, sz, down, opts, varargin{:}) ;
  net.name = [name '_conv'] ;

  bn = opts.modelOpts.batchNormalization ;
  rn = opts.modelOpts.batchRenormalization ;
  assert(bn + rn < 2, 'cannot add both batch norm and renorm') ;
  if bn
    net = vl_nnbnorm(net, 'learningRate', [2 1 0.05], 'testMode', false) ;
    net.name = [name '_bn'] ;
  elseif rn
    net = vl_nnbrenorm_auto(net, opts.clips, opts.renormLR{:}) ; 
    net.name = [name '_rn'] ;
  end

  if useRelu
    net = vl_nnrelu(net) ;
    net.name = [name '_relu'] ;
  end

% ----------------------------------------------------
function net = add_conv(net, sz, down, opts, varargin)
% ----------------------------------------------------
  filters = Param('value', init_weight(sz, 'single'), 'learningRate', 1) ;
  biases = Param('value', zeros(sz(4), 1, 'single'), 'learningRate', 2) ;
  if down, stride = 2 ; else, stride = 1 ; end
  args = {'pad', (sz(1) - 1) / 2, 'stride', stride, ...
          'CudnnWorkspaceLimit', opts.modelOpts.CudnnWorkspaceLimit} ;
  net = vl_nnconv(net, filters, biases, args{:}, varargin{:}) ;

% --------------------------------------
function weights = init_weight(sz, type)
% --------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

  sc = sqrt(2/(sz(1)*sz(2)*sz(4))) ;  
  weights = randn(sz, type)*sc ;
