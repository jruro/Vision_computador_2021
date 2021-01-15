%% Sign Alphabet Classification

clearvars
clear all
clc

%% INICIALIZACIÓN, CARGA DEL DATASET

% Se establece la direccion del dataset
dir = './Gesture Image Data/';

% Se carga el dataset completo
allData = imageDatastore(dir, 'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');

% Se seleccionan los datos a clasificar
%[Data, noData] = splitEachLabel(allData,0.6,'Randomized');
Data = allData;
[trainingSet, testSet] = splitEachLabel(Data,0.6,'Randomized'); % training set es el de prueba y el test set es el de entrenamiento

% Se muestran los datos a clasificar
countEachLabel(trainingSet);
countEachLabel(testSet);

% Se toma imagen de prueba para sacar tamaño de la matrices
img = readimage(trainingSet, 1);

% Get labels for each image.
trainingLabels = trainingSet.Labels;
testLabels = testSet.Labels;

%% DESCRIPTORES ENTRENAMIENTO

tipo = 1;

% Parámetros del árbol de decisión
Y = [trainingLabels];
X = helperExtractFeatures(trainingSet,testSet,tipo);

% Árbol de decicision con el entrenamiento
Mdl = fitctree(X,Y);
Mdl_c = compact(Mdl);

predictedLabels = predict(Mdl_c, X);

% Ploteo del resultado del entrenamiento
figure(1)
cm = confusionchart(trainingLabels,predictedLabels,...
    'RowSummary','row-normalized','ColumnSummary','column-normalized');

%% DESCRIPTORES TESTEO

tipo = 0;

% Parámetros del árbol de decisión
Ytest = [testLabels];
Xtest = helperExtractFeatures(trainingSet,testSet,tipo);

% Árbol de decicision del testeo
Mdl_test = fitctree(Xtest,Ytest);
Mdl_c_test = compact(Mdl_test);

predictedLabels = predict(Mdl_c_test, Xtest);

% Ploteo del resultado del testeo
figure(2)
cm2 = confusionchart(testLabels,predictedLabels,...
    'RowSummary','row-normalized','ColumnSummary','column-normalized');

%% PRUEBAS

% XtestFeatureSize = length(Xtest);

% for k = 1:50
%     i = randi([1 55500]);
%     
%     img = readimage(allData, i);
%     imgFeatures(1, :) = helperExtractFeatures(trainingSet);
%     
%     trainingLabels = trainingSet.Labels;
%     testLabel = trainingLabels(i);
%     
%     predictedLabels = predict(classifier, imgFeatures);
%     
%     figure(2)
%     subplot(5,10,k)
%     imshow(img)
%     title(['Entrada: ' testLabel ' Prediccion: ' predictedLabels])
%     
%     TESTLABELS(k) = testLabel;
%     PREDICTEDLABELS(k) = predictedLabels;
% end
