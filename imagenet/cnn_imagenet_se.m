function [net, info] = cnn_imagenet_se(varargin)
%CNN_IMAGENET_SE Train a Squeeze-and-Excitation CNN on ImageNet
%  CNN_IMAGENET_SE trains an SENet from scratch on the ImageNet
%  2012 training data.
%
%  based on the cnn_imagenet.m MatConvNet example script by 
%  Andrea Vedaldi
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.gpus = 4 ;
  opts.lite = false ;
  opts.numFetchThreads = 12 ;
  opts.modelType = 'alexnet' ;
  opts.networkType = 'simplenn' ;
  opts.batchNormalization = true ;
  opts.weightInitMethod = 'gaussian' ;
  opts.dataDir = fullfile(vl_rootnn, 'data','datasets', 'ILSVRC2012') ;
  opts.expDir = fullfile(vl_rootnn, 'data', 'senets-imagenet12') ;
  opts.imdbPath = fullfile(vl_rootnn, 'data', 'imagenet12', 'imdb.mat');
  opts = vl_argparse(opts, varargin) ;

  opts.train.gpus = opts.gpus ; % train opts

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

  if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
    imdb.imageDir = fullfile(opts.dataDir, 'images');
  else
    imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
  end

  % Compute image statistics (mean, RGB covariances, etc.)
  if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
  imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
  if exist(imageStatsPath, 'file')
    load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
  else
    train = find(imdb.images.set == 1) ;
    images = fullfile(imdb.imageDir, imdb.images.name(train(1:100:end))) ;
    [averageImage, rgbMean, rgbCovariance] = getImageStats(images, ... 
                                       'imageSize', [256 256], ...
                                       'numThreads', opts.numFetchThreads, ...
                                       'gpus', opts.train.gpus) ; %#ok
    save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
  end
  [v,d] = eig(rgbCovariance) ; rgbDeviation = v*sqrt(d) ; clear v d ;

  % set model options
  modelOpts.batchNormalization = true ; 
  modelOpts.batchRenormalization = false ; 
  modelOpts.averageImage = rgbMean ;
  modelOpts.colorDeviation = rgbDeviation ;
  modelOpts.classNames = imdb.classes.name ;
  modelOpts.classDescriptions = imdb.classes.description ;
  modelOpts.CudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
  opts.modelOpts = modelOpts ;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

  net = cnn_imagenet_init_senet(opts) ;
  %'averageImage', rgbMean, ...
                                %'colorDeviation', rgbDeviation, ...
                                %'classNames', imdb.classes.name, ...
                                %'classDescriptions', imdb.classes.description) ;

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

  [net, info] = cnn_train_dag(net, imdb, getBatchFn(opts, net.meta), ...
                        'expDir', opts.expDir, ...
                        net.meta.trainOpts, ...
                        opts.train) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------

  net = cnn_imagenet_deploy(net) ;
  modelPath = fullfile(opts.expDir, 'net-deployed.mat') ;
  net_ = net.saveobj() ; save(modelPath, '-struct', 'net_') ; clear net_ ; %#ok

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------

  if numel(meta.normalization.averageImage) == 3
    mu = double(meta.normalization.averageImage(:)) ;
  else
    mu = imresize(single(meta.normalization.averageImage), ...
                  meta.normalization.imageSize(1:2)) ;
  end
  useGpu = numel(opts.train.gpus) > 0 ;
  bopts.test = struct(...
    'useGpu', useGpu, ...
    'numThreads', opts.numFetchThreads, ...
    'imageSize',  meta.normalization.imageSize(1:2), ...
    'cropSize', meta.normalization.cropSize, ...
    'subtractAverage', mu) ;

  % Copy the parameters for data augmentation
  bopts.train = bopts.test ;
  fnames = fieldnames(meta.augmentation) ;
  for ii = 1:numel(fnames)
    bopts.train.(fnames{ii}) = meta.augmentation.(fnames{ii}) ;
  end
  fn = @(x,y) getBatch(bopts,useGpu,lower(opts.networkType),x,y) ;

% -------------------------------------------------------------------------
function varargout = getBatch(opts, useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
  images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
  if ~isempty(batch) && imdb.images.set(batch(1)) == 1
    phase = 'train' ;
  else
    phase = 'test' ;
  end
  data = getImageBatch(images, opts.(phase), 'prefetch', nargout == 0) ;
  if nargout > 0
    labels = imdb.images.label(batch) ;
    varargout{1} = {'input', data, 'label', labels} ;
  end
