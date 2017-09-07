function add_se_meta_info(varargin)
%ADD_SE_META_INFO - adds additional meta information to imported SE models
%   ADD_SE_META_INFO adds informaiton about the imagenet dataset used for 
%   training to each model to facilitate easier use in deployment
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
  opts.imdbPath = fullfile(vl_rootnn, 'data/imagenet12/imdb.mat') ;
  opts = vl_argparse(opts, varargin) ;

  imdb = load(opts.imdbPath) ;
  res = dir(fullfile(opts.modelDir, '*.mat')) ; modelNames = {res.name} ;
  modelNames = modelNames(contains(modelNames, 'SE')) ;
  
  for ii = 1:numel(modelNames)
    modelPath = fullfile(opts.modelDir, modelNames{ii}) ;
    fprintf('adding info to %s (%d/%d)\n', modelPath, ii, numel(modelNames)) ;
    net = load(modelPath) ;
    net.meta.classes = imdb.classes ;
    save(modelPath, '-struct', 'net') ;
  end

