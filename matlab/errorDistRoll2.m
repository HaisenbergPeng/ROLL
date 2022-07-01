clc
clear;
close all
%% Sometimes the code malfunctions and you will get wierd statistics;
% don't know why so just reboot the MATLAB
% folder = "/mnt/sdb/Datasets/NCLT/datasets/fastlio_noTMM";
folder = "/mnt/sdb/Datasets/NCLT/datasets/logs/roll2";
% folder = "/mnt/sdb/Datasets/NCLT/datasets/no_LIO";
% folder = "/mnt/sdb/Datasets/NCLT/datasets/LOAM";

date = "2012-02-02";
% dateS = "2013-02-23withDownsamplePC";
logFilePath = folder+"/"+date+"-0.5HZ/map_pcd/mappingError.txt";
% poseFilePath = folder+"/"+date+"/map_pcd/path_mapping.txt";
% poseFilePath = folder+"/"+date+"-0.5HZ/map_pcd/path_vinsfusion.txt";
poseFilePath = folder+"/"+date+"-0.5HZ/map_pcd/path_fusion.txt";
gtFilePath = "/mnt/sdb/Datasets/NCLT/datasets/ground_truth/groundtruth_"+date+".csv";


% %% log file reading
% fID = fopen(logFilePath);
% strPattern = "";
% n = 11;
% for i=1:n
%     strPattern = strPattern+"%f";
% end
% logData = textscan(fID,strPattern);
% timeLog = logData{1}-logData{1}(1);
% regiError = logData{5};
% inlierRatio2 = logData{4};
% inlierRatio = logData{3};
% isTMM = logData{2};

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

%% gt reading
% readcsv readmatrix:sth is wrong
fID3 = fopen(gtFilePath);
gtData = textscan(fID3, "%f%s%f%s%f%s%f%s%f%s%f%s%f");
% downsample
downsample = 10;
lenGT = length(gtData{1});
matGT = zeros(floor(lenGT/downsample),7);
%% ROLL uses imu pose, so here convert body pose to imu pose
tbi = [-0.11 -0.18 -0.71]';
for i=1:floor(lenGT/downsample)
    for j=1:7
        matGT(i,j) = gtData{2*j-1}(downsample*i);
    end
    Rmb = eul2rotm([ matGT(i,7),matGT(i,6),matGT(i,5)],"ZYX");
    tmb = matGT(i,2:4)';
    tmi = Rmb*tbi +tmb;
    matGT(i,2:4) = tmi';
end

%% sync with time
timeGT = matGT(:,1)/1e+6; % us -> sec
timePose =  matPose(:,1)/1e+6;
% timeGT = timeGT -timeGT(1);
% timePose = timePose - timePose(1);
% MDtimeGT = KDTreeSearcher(timeGT);
[idx, D] = rangesearch(timeGT,timePose,0.05);
ateErrorINI = zeros(lenPose,1);
Err = zeros(lenPose,6);
not_found = 0;
idxC = 0;
for i=1:lenPose
    if isempty(idx{i})
        not_found = not_found + 1;
        continue;    
    end
        %% rule out obvious wrong ground truth
    if date=="2013-02-23" && matPose(i,2)>-310 && matPose(i,2)<-260 &&...
        matPose(i,3)>-450 && matPose(i,3)<-435
        continue;
    end
    idxC = idxC + 1;
%     ateError(i) = norm(matPose(i,2:3)-matGT(idx{i}(1),2:3));
    Err(i,:) = matPose(i,2:7)-matGT(idx{i}(1),2:7);
    for iE = 1:3
        Err(i,iE+3) = 180/pi*(Err(i,iE+3) - 2*pi*round(Err(i,iE+3)/2/pi));
    end
    deltaT = transError(matGT(idx{i}(1),2:7),matPose(i,2:7));
    ateErrorINI(idxC) = norm(deltaT(1:3,4));

end
ateError = ateErrorINI(1:idxC);

disp("RMSE error: "+norm(ateError)/sqrt(idxC))
disp("max error: "+max(ateError))
disp("Loc rate: "+length(ateError)/(timePose(end)-timePose(1)))
disp("Success ratio: "+100*length(find(ateError < 1.0))/(timeGT(end)-timeGT(1))/10)
disp("<0.1 %: "+ 100*length(find(ateError < 0.1))/idxC)
disp("<0.2 %: "+ 100*length(find(ateError < 0.2))/idxC)
disp("<0.5 %: "+ 100*length(find(ateError < 0.5))/idxC)
disp("<1.0 %: "+ 100*length(find(ateError < 1.0))/idxC)

figure(1)
hold on
plot(matPose(:,2),matPose(:,3));
plot(matGT(:,2),matGT(:,3));
% legend("LOAM(M)+TM","LOAM(M)","ROLL","G.T.");
legend("ROLL","G.T.");
xlabel("X (m)");
ylabel("Y (m)");

figure(2)
subplot(3,2,1)
plot(timePose-timePose(1),Err(:,1),'r');
xlabel("Time (s)");
ylabel("Error in x (m)");
subplot(3,2,3)
plot(timePose-timePose(1),Err(:,2),'r');
xlabel("Time (s)");
ylabel("Error in y (m)");
subplot(3,2,5)
plot(timePose-timePose(1),Err(:,3),'r');
xlabel("Time (s)");
ylabel("Error in z (m)");

subplot(3,2,2)
plot(timePose-timePose(1),Err(:,4),'r');
xlabel("Time (s)");
ylabel("Error in roll (^{\circ})");
subplot(3,2,4)
plot(timePose-timePose(1),Err(:,5),'r');
xlabel("Time (s)");
ylabel("Error in pitch (^{\circ})");
subplot(3,2,6)
plot(timePose-timePose(1),Err(:,6),'r');
xlabel("Time (s)");
ylabel("Error in yaw (^{\circ})");
legend("LOAM(M)","LOAM(M)+TM","ROLL");


% figure(3)
% hold on
% histogram(ateError)
% legend("LOAM(M)","LOAM(M)+TM","ROLL");
% xlabel("Translational distance error");
% ylabel("Count");
function eT = transError(Vgt,V2)
% input: x y z r p y
    T1 = eye(4);
    T2 = eye(4);
    T1(1:3,1:3) = eul2rotm([Vgt(6),Vgt(5),Vgt(4)],"ZYX");
    T2(1:3,1:3) = eul2rotm([V2(6),V2(5),V2(4)],"ZYX");
    T1(1:3,4) = Vgt(1:3)';
    T2(1:3,4) = V2(1:3)';
    eT = T1\T2;    % equals to T1\T2
end

function y = removeJump(x)
    % 2pi
    Ntmp = round(x/360);
    y = x - Ntmp*360;
end