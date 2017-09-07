function run_se_benchmarks
% RUN_SE_BENCHMARKS do a single pass over the imagenet validation data
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

gpus = 1 ;
batchSize = 32 ;
useCached = 0 ; % load results from cache if available

importedModels = {
'SE-ResNet-50-mcn', ...
} ;

for ii = 1:numel(importedModels)
  model = importedModels{ii} ;
  imagenet_eval(model, batchSize, gpus, useCached) ;
end

% -------------------------------------------------------
function imagenet_eval(model, batchSize, gpus, useCached)
% -------------------------------------------------------
[~,info] = cnn_imagenet_se_mcn('model', model, 'batchSize', ...
               batchSize, 'gpus', gpus, 'continue', useCached) ;
top1 = info.val.top1err * 100 ; top5 = info.val.top5err * 100 ;
fprintf('%s: top-1: %.2f, top-5: %.2f\n', model, top1, top5) ;
