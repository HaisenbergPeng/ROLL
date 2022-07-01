clc;
clear;
close all
% folder = "/media/haisenberg/BIGLUCK/Datasets/NCLT/datasets/fastlio_noTMM";
% folder = "/media/haisenberg/BIGLUCK/Datasets/NCLT/datasets/fastlio_loc2";
folder = "/media/haisenberg/BIGLUCK/Datasets/NCLT/datasets/LOAM_TM";
folderL = "/media/haisenberg/BIGLUCK/Datasets/NCLT/datasets/LOAM";

date = "2012-05-11";
logFilePath = folder+"/"+date+"/map_pcd/mappingError.txt";
% poseFilePath = folder+"/"+date+"_bin/map_pcd/path_mapping.txt";
% poseFilePathL = folderL+"/"+date+"_bin/map_pcd/path_mapping.txt";
poseFilePath = folder+"/"+date+"_bin/map_pcd/path_fusion.txt";
poseFilePathL = folderL+"/"+date+"_bin/map_pcd/path_fusion.txt";
gtFilePath = "/media/haisenberg/BIGLUCK/Datasets/NCLT/datasets/"+date+"/groundtruth_"+date+".csv";

%% pose file reading
fID2 = fopen(poseFilePath);
strPattern = "";
n = 7;
for i=1:n
    strPattern = strPattern+"%f";
end
poseData = textscan(fID2,strPattern);
lenPose = length(poseData{1});
matPose = zeros(lenPose,7);
for i=1:lenPose
    for j=1:7
        matPose(i,j) = poseData{j}(i);
    end
end
timePose =  matPose(:,1)/1e+6;

%% pose file reading 2
fID2 = fopen(poseFilePathL);
strPattern = "";
n = 7;
for i=1:n
    strPattern = strPattern+"%f";
end
poseData = textscan(fID2,strPattern);
lenPoseL = length(poseData{1});
matPoseL = zeros(lenPoseL,7);
for i=1:lenPoseL
    for j=1:7
        matPoseL(i,j) = poseData{j}(i);
    end
end
timePoseL =  matPoseL(:,1)/1e+6;

%% gt reading
% readcsv readmatrix:sth is wrong
fID3 = fopen(gtFilePath);
gtData = textscan(fID3, "%f%s%f%s%f%s%f%s%f%s%f%s%f");
% downsample
downsample = 10;
lenGT = length(gtData{1});
matGT = zeros(floor(lenGT/10),7);
%% LOAM uses lidar pose, so here convert body pose to lidar pose
% tbi = [-0.11 -0.18 -0.71]';
for i=1:floor(lenGT/10)
    for j=1:7
        matGT(i,j) = gtData{2*j-1}(10*i);
    end
    matGT(i,2:7) = body2lidar(matGT(i,2:7)); % why no need?
end


timeGT = matGT(:,1)/1e+6; % us -> sec
MDtimeGT = KDTreeSearcher(timeGT);
[idx, D] = rangesearch(MDtimeGT,timePose,0.05);
not_found = 0;
idxC = 0;
%% sync with time 
ateErrorINI = zeros(lenPose,1);
Err = zeros(lenPose,6);
for i=1:lenPose
    if isempty(idx{i})
        not_found = not_found + 1;
        continue;    
    end
    idxC = idxC + 1;
    Err(i,:) = matPose(i,2:7)-matGT(idx{i}(1),2:7);
    for iE = 1:3
        Err(i,iE+3) = 180/pi*(Err(i,iE+3) - 2*pi*round(Err(i,iE+3)/2/pi));
    end
    deltaT = transError(matGT(idx{i}(1),2:7),matPose(i,2:7));
    ateErrorINI(idxC) = norm(deltaT(1:3,4));  
end
ateError = ateErrorINI(1:idxC);
%% sync with time 2
[idxL, D] = rangesearch(MDtimeGT,timePoseL,0.05);
ateErrorINIL = zeros(lenPoseL,1);
ErrL = zeros(lenPoseL,6);
idxC = 0;
for i=1:lenPoseL
    if isempty(idxL{i})
        not_found = not_found + 1;
        continue;    
    end 
    idxC = idxC + 1;
    ErrL(i,:) = matPoseL(i,2:7)-matGT(idxL{i}(1),2:7);
    for iE = 1:3
        ErrL(i,iE+3) = 180/pi*(ErrL(i,iE+3) - 2*pi*round(ErrL(i,iE+3)/2/pi));
    end
    deltaT = transError(matGT(idxL{i}(1),2:7),matPose(i,2:7));
    ateErrorINIL(idxC) = norm(deltaT(1:3,4));  
end
ateErrorL = ateErrorINIL(1:idxC);
disp("----------------LOAM----------------")
disp("RMSE error: "+norm(ateErrorL)/sqrt(length(ateErrorL)))
disp("max error: "+max(ateErrorL))
disp("Loc rate: "+length(ateErrorL)/(timePoseL(end)-timePoseL(1)))
disp("Success ratio %: "+100*length(find(ateErrorL < 1.0))/(timeGT(end)-timeGT(1))/10)
disp("<0.1 %: "+ 100*length(find(ateErrorL < 0.1))/length(ateErrorL))
disp("<0.2 %: "+ 100*length(find(ateErrorL < 0.2))/length(ateErrorL))
disp("<0.5 %: "+ 100*length(find(ateErrorL < 0.5))/length(ateErrorL))
disp("<1.0 %: "+ 100*length(find(ateErrorL < 1.0))/length(ateErrorL))
disp("----------------LOAM+TM----------------")
disp("RMSE error: "+norm(ateError)/sqrt(length(ateError)))
disp("max error: "+max(ateError))
disp("Loc rate: "+length(ateError)/(timePose(end)-timePose(1)))
disp("Success ratio %: "+100*length(find(ateError < 1.0))/(timeGT(end)-timeGT(1))/10)
disp("<0.1 %: "+ 100*length(find(ateError < 0.1))/length(ateError))
disp("<0.2 %: "+ 100*length(find(ateError < 0.2))/length(ateError))
disp("<0.5 %: "+ 100*length(find(ateError < 0.5))/length(ateError))
disp("<1.0 %: "+ 100*length(find(ateError < 1.0))/length(ateError))


figure(1)
plot(matPoseL(:,2),matPoseL(:,3),'b');
hold on
plot(matPose(:,2),matPose(:,3),'Color',[0.9290 0.6940 0.1250]);

figure(2)
subplot(3,2,1)
plot(timePoseL-timePoseL(1),ErrL(:,1),'b');
hold on
plot(timePose-timePose(1),Err(:,1),'Color',[0.9290 0.6940 0.1250]);

subplot(3,2,3)
plot(timePoseL-timePoseL(1),ErrL(:,2),'b');
hold on
plot(timePose-timePose(1),Err(:,2),'Color',[0.9290 0.6940 0.1250]);

subplot(3,2,5)
plot(timePoseL-timePoseL(1),ErrL(:,3),'b');
hold on
plot(timePose-timePose(1),Err(:,3),'Color',[0.9290 0.6940 0.1250]);

subplot(3,2,2)
plot(timePoseL-timePoseL(1),ErrL(:,4),'b');
hold on
plot(timePose-timePose(1),Err(:,4),'Color',[0.9290 0.6940 0.1250]);

subplot(3,2,4)
plot(timePoseL-timePoseL(1),ErrL(:,5),'b');
hold on
plot(timePose-timePose(1),Err(:,5),'Color',[0.9290 0.6940 0.1250]);

subplot(3,2,6)
plot(timePoseL-timePoseL(1),ErrL(:,6),'b');
hold on
plot(timePose-timePose(1),Err(:,6),'Color',[0.9290 0.6940 0.1250]);

% figure(3)
% histogram(ateErrorL)
% hold on
% histogram(ateError)


function eT = transError(Vgt,V2)
% input: x y z r p y
    T1 = eye(4,4);
    T2 = eye(4,4);
    T1(1:3,1:3) = eul2rotm([Vgt(6),Vgt(5),Vgt(4)],"ZYX");
    T2(1:3,1:3) = eul2rotm([V2(6),V2(5),V2(4)],"ZYX");
    T1(1:3,4) = Vgt(1:3);
    T2(1:3,4) = V2(1:3);
    eT = inv(T1)*T2;    
end

function Vlidar = body2lidar(Vbody)
% input and output: 1x6
    Tmb = eye(4,4);
    Tmb(1:3,1:3) = eul2rotm([Vbody(6),Vbody(5),Vbody(4)],"ZYX");
    Tmb(1:3,4) = Vbody(1:3)';
    tlb = [0.002, -0.004, -0.957];
    eulLB = [0.014084807063594,0.002897246558311,-1.583065991436417];
    Tbl = eye(4,4);
    Tbl(1:3,4) = tlb';
    Tbl(1:3,1:3) = eul2rotm([eulLB(3),eulLB(2),eulLB(1)],"ZYX");
    Tml= Tmb*Tbl;
    Vlidar = zeros(1,6);
    Vlidar(1:3) = Tml(1:3,4)';
    tmp = rotm2eul(Tml(1:3,1:3),"ZYX");
    Vlidar(4:6) = [tmp(3),tmp(2),tmp(1)];
end

function y = removeJump(x)
    N = round(x/pi/2);
    y = x-N*pi*2;
end
