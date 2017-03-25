%read the images ignoring the header
Tcsv = csvread('train.csv',1,0);
TstCsv = csvread('test.csv',1,0);

%separe the labels and the images
labels = Tcsv(:,1);
Tcsv = Tcsv(:,2:785);

parpool; %following operations are doing in parallel.

TrainImages = zeros(28,28,42000);
TestImages = zeros(28,28,28000);

%move images from interval [0 255] to [0 1]
parfor i = 1:42000
    TrainImages(:,:,i) = im2double(reshape(Tcsv(i,:),28,28));
end

parfor i = 1:28000
    TestImages(:,:,i) = im2double(reshape(TstCsv(i,:),28,28));
end

%several algorithms needs the binary images to work, we can use them too. 
BinaryTrainImages = zeros(28,28,42000);
BinaryTestImages =  zeros(28,28,28000);

parfor i = 1:42000
BinaryTrainImages(:,:,i) = im2bw(TrainImages(:,:,i),0.5);
end

parfor i = 1:28000
BinaryTestImages(:,:,i) = im2bw(TestImages(:,:,i),0.5);
end

%erode
se = strel('disk',2); % erode filter, disk is used since ball-pen is suposed to be used for image
%2 pixel filter is used to avoid eliminate too much pixels

parfor i = 1:42000
erodeTrain(:,:,i) = imerode(BinaryTrainImages(:,:,i),se);
end

parfor i= 1:28000
erodeTest(:,:,i) = imerode(BinaryTestImages(:,:,i),se);
end

%close
closeTrain = zeros(28,28,42000);
closeTest = zeros(28,28,28000);

parfor i = 1:42000
closeTrain(:,:,i) = imdilate(BinaryTrainImages(:,:,i),se);
end

parfor i = 1:28000
closeTest(:,:,i) = imdilate(BinaryTestImages(:,:,i),se);
end

%canny
cannyTrain = zeros(28,28,42000);
cannyTest = zeros(28,28,28000);

parfor i = 1:42000
cannyTrain(:,:,i) = edge(BinaryTrainImages(:,:,i),'canny');
end

parfor i = 1:28000
cannyTest(:,:,i) = edge(BinaryTestImages(:,:,i),'canny');
end


%sobel-gradient
sobelTrain =  zeros(28,28,42000);
sobelTest = zeros(28,28,28000);

parfor i = 1:42000
   [~, mg] = imgradient(TrainImages(:,:,i));
   sobelTrain(:,:,i) = mg;
end

parfor i = 1:28000
   [~, mg] = imgradient(TestImages(:,:,i));
   sobelTest(:,:,i) = mg;
end

%deshacer el tama?o de im?genes

BinaryCSV = zeros(42000,785);
for i = 1:42000
    BinaryCSV(i,2:785) = reshape(BinaryTrainImages(:,:,1),1,784);
    BinaryCSV(i,1) = labels(i);
end
csvwrite('binary-train.csv',BinaryCSV);
%TODO test saving

%Skeleton

SkelCSV = zeros(42000,785);
for i = 1:42000
    SkelCSV(i,2:785) = reshape(erodeTrain(:,:,1),1,784);
    SkelCSV(i,1) = labels(i);
end
csvwrite('skel-train.csv',SkelCSV);

SkelCSV = zeros(28000,784);
parfor i = 1:28000
    SkelCSV(i,:) = reshape(erodeTest(:,:,1),1,784);
end
csvwrite('skel-test.csv',SkelCSV);

%Close

ClosedCSV = zeros(42000,785);
for i = 1:42000
    ClosedCSV(i,2:785) = reshape(closeTrain(:,:,1),1,784);
    ClosedCSV(i,1) = labels(i);
end
csvwrite('closed-train.csv',ClosedCSV);

ClosedCSV = zeros(28000,784);
parfor i = 1:28000
    ClosedCSV(i,:) = reshape(closeTest(:,:,1),1,784);
end
csvwrite('closed-test.csv',ClosedCSV);

%canny

CannyCSV = zeros(42000,785);
for i = 1:42000
    CannyCSV(i,2:785) = reshape(cannyTrain(:,:,1),1,784);%% !
    CannyCSV(i,1) = labels(i);
end
csvwrite('canny-train.csv',CannyCSV);

CannyCSV = zeros(28000,784);
parfor i = 1:28000
    CannyCSV(i,:) = reshape(cannyTest(:,:,1),1,784);
end
csvwrite('canny-test.csv',CannyCSV);


%Gradient

GradientCSV = zeros(42000,785);
for i = 1:42000
    GradientCSV(i,2:785) = reshape(sobelTrain(:,:,1),1,784);%% !
    GradientCSV(i,1) = labels(i);
end
csvwrite('gradient-train.csv',GradientCSV);


GradientCSV = zeros(28000,784);
parfor i = 1:28000
    GradientCSV(i,:) = reshape(sobelTest(:,:,1),1,784);
end
csvwrite('gradient-test.csv',GradientCSV);
