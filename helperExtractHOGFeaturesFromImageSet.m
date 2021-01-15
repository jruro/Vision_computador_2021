function [features, setLabels] = helperExtractHOGFeaturesFromImageSet(imds, hogFeatureSize1, cellSize)
% Extract HOG features from an imageDatastore.

setLabels = imds.Labels;
numImages = numel(imds.Files);
features  = zeros(numImages, hogFeatureSize1, 'single');

% Process each image and extract features
for j = 1:numImages
    img = readimage(imds, j);
    features(j, :) = extractHOGFeatures(img,'CellSize',cellSize);
end
end