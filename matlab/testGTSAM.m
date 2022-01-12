clc;
close all;
clear
date = "2012-02-02";
folder = "/media/haisenberg/BIGLUCK/Datasets/NCLT/datasets/fastlio_loc2";
poseFilePath = folder+"/"+date+"/map_pcd/path_mapping.txt";
poseFilePath2 = folder+"/"+date+"/map_pcd/path_mappingOPT.txt";
%% pose file reading
fID = fopen(poseFilePath);
strPattern = "";
n = 7;
for i=1:n
    strPattern = strPattern+"%f";
end
poseData = textscan(fID,strPattern);
% lenPose = length(poseData{1});
% matPose = zeros(lenPose,7);
% for i=1:lenPose
%     for j=1:7
%         matPose(i,j) = poseData{j}(i);
%     end
% end
fID2 = fopen(poseFilePath2);
poseData2 = textscan(fID2,strPattern);

figure(1)
plot(poseData{2},poseData{3});
hold on
plot(poseData2{1},poseData2{2});