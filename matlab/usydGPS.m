clc;
clear;
close all
%% icp registration for adjustment
gpsFile = "/mnt/sdb/Datasets/Usyd/MAPs/2018-03-08/map_pcd/gps.txt";
poseFilePath = "/mnt/sdb/Datasets/Usyd/MAPs/2018-03-08/map_pcd/path_vinsfusion.txt";
fID = fopen(gpsFile);
strPattern = "";
n = 7;
for i=1:n
    strPattern = strPattern+"%f";
end
gpsData = textscan(fID,strPattern);
pGPS = [gpsData{2},gpsData{3},gpsData{4}];



%% usyd mapping
fID = fopen(poseFilePath);
strPattern = "";
n = 11;
for i=1:n
    strPattern = strPattern+"%f";
end
poseData = textscan(fID,strPattern);
timeLog = poseData{1}-poseData{1}(1);
pROLL = [poseData{2},poseData{3},poseData{4}];

figure(1)
plot(pGPS(:,1),pGPS(:,2))
hold on
plot(pROLL(:,1),pROLL(:,2));

%% solve it numerically: using the first 100 corresponding frames
T

% %% icp registration: transform the former to the latter
% %% no good
% R = [0.948625333734415,0.180415552350336,0.259923459247897;
%     -0.173974299683239,0.983590472604959,-0.0477778740682419;
%     -0.264278109662902,0.000103299939208878,0.964446509704984];
% Tinit = rigid3d(R,zeros(1,3));
% pcGPS = pointCloud(pGPS);
% pcROLL = pointCloud(pROLL);
% [T,pcROLLout] = pcregistericp(pcGPS,pcROLL,"InitialTransform",Tinit);
% 
% pGPSnew = convertT(pGPS,T);
% 
% plot(pGPSnew(:,1),pGPSnew(:,2))
% legend("GPS no aligned","ROLL poses ","GPS aligned");
% 
% figure(2)
% pcshow(pcROLLout)
% hold on
% pcshow(pcGPS);
% function pO = convertT(p,T)
%     rot = T.Rotation;
%     trans = T.Translation;
%     pO = rot'*p'+trans';
%     pO = pO';
% end