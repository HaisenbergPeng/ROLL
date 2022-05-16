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

%% icp registration: transform the former to the latter
pcGPS = pointCloud(pGPS);
pcROLL = pointCloud(pROLL);
[T,pcROLLout] = pcregistericp(pcGPS,pcROLL);

pGPSnew = convertT(pGPS,T);

plot(pGPSnew(:,1),pGPSnew(:,2))
legend("GPS no aligned","ROLL poses ","GPS aligned");

figure(2)
pcshow(pcROLLout)
hold on
pcshow(pcGPS);
function pO = convertT(p,T)
    rot = T.Rotation;
    trans = T.Translation;
    pO = rot'*p'+trans';
    pO = pO';
end