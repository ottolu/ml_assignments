function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%     

numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);

poolSize = floor(convolvedDim / poolDim);
pooledFeatures = zeros(numFeatures, numImages, poolSize, poolSize);

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------

for i = 1 : numImages
   for j = 1 :  numFeatures
       for m = 1 : poolSize
           for n = 1 : poolSize
               rowEnd = poolDim * m;
               rowBegin = rowEnd - poolDim + 1;
               colEnd = poolDim * n;
               colBegin = colEnd - poolDim + 1;
               pooledFeatures(j, i, m, n) = ...
                    mean(mean(convolvedFeatures(j, i, rowBegin:rowEnd, colBegin:colEnd)));
           end
       end
   end
end

end

