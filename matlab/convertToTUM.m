clc;
clear;
close all
%% icp registration for adjustment
gpsFile = "/mnt/sdb/Datasets/Usyd/MAPs/2018-03-08/map_pcd/gps.txt";
poseFilePath = "/mnt/sdb/Datasets/Usyd/MAPs/2018-03-08/map_pcd/path_vinsfusion.txt";
outputDir = "/mnt/sdb/Datasets/Usyd/MAPs/2018-03-08/EVO";
fID = fopen(gpsFile);
strPattern = "";
n = 7;
for i=1:n
    strPattern = strPattern+"%f";
end
gpsData = textscan(fID,strPattern);

lenGPS = length(gtData{1});
poseGPS = zeros(lenGPS,8);
fIDgps = fopen(outputDir+"/gps_evo.txt");
for i=1:lenGPS
    eulGPS = [gpsData{4}(i),gpsData{3}(i),gpsData{2}(i)];
    quat = eul2quat(eulGPS,"ZYX"); % wxyz
    fprintf(fIDgps,"",[]);
end
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