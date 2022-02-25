% clc;
clear;
close all
% folder = "/media/haisenberg/BIGLUCK/Datasets/NCLT/datasets/fastlio_noTMM";
% folder = "/media/haisenberg/BIGLUCK/Datasets/NCLT/datasets/fastlio_loc2";
folder = "/media/haisenberg/BIGLUCK/Datasets/NCLT/datasets/no_LIO";
% folder = "/media/haisenberg/BIGLUCK/Datasets/NCLT/datasets/LOAM";

date = "2012-05-11";
logFilePath = folder+"/"+date+"/map_pcd/mappingError.txt";
poseFilePath = folder+"/"+date+"_bin/map_pcd/path_mapping.txt";
% poseFilePath = folder+"/"+date+"/map_pcd/path_vinsfusion.txt";
% poseFilePath = folder+"/"+date+"_bin/map_pcd/path_fusion.txt";
gtFilePath = "/media/haisenberg/BIGLUCK/Datasets/NCLT/datasets/"+date+"/groundtruth_"+date+".csv";

%% log file reading
fID = fopen(logFilePath);
strPattern = "";
n = 11;
for i=1:n
    strPattern = strPattern+"%f";
end
logData = textscan(fID,strPattern);
timeLog = logData{1}-logData{1}(1);
regiError = logData{3};
inlierRatio2 = logData{3};
inlierRatio = logData{2};
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
matGT = zeros(floor(lenGT/10),7);
%% LOAM uses lidar pose, so here convert body pose to lidar pose
% tbi = [-0.11 -0.18 -0.71]';
for i=1:floor(lenGT/10)
    for j=1:7
        matGT(i,j) = gtData{2*j-1}(10*i);
    end
%     matGT(i,2:7) = body2lidar(matGT(i,2:7)); % why no need?
end

%% sync with time
timeGT = matGT(:,1)/1e+6; % us -> sec
timePose =  matPose(:,1)/1e+6;
MDtimeGT = KDTreeSearcher(timeGT);
[idx, D] = rangesearch(MDtimeGT,timePose,0.05);
ateErrorINI = zeros(lenPose,1);
yawError = zeros(lenPose,1);
not_found = 0;
idxC = 0;
for i=1:lenPose
    if isempty(idx{i})
        not_found = not_found + 1;
        continue;    
    end
        %% rule out obvious wrong ground truth
    if date=="2013-02-23" && matPose(i,2)>-310 && matPose(i,2)<-260&&...
        matPose(i,3)>-450 && matPose(i,3)<-435
        continue;
    end
    idxC = idxC + 1;
%     ateError(i) = norm(matPose(i,2:3)-matGT(idx{i}(1),2:3));
%     yawError(i) = 180/pi*(matPose(i,7)-matGT(idx{i}(1),7));
    deltaT = transError(matGT(idx{i}(1),2:7),matPose(i,2:7));
    ateErrorINI(idxC) = norm(deltaT(1:3,4));
    %% 2pi
    Ntmp = round(yawError(i)/360);
    yawError(i) = yawError(i) - Ntmp*360;
%     if ateError(i)>1
%         matGT(idx{i}(1),2:3)
%     end
end
ateError = ateErrorINI(1:idxC);
idxOver1m= find(ateError > 1.0);
%% PLOT
figure(1)
plot(timePose-timePose(1),ateErrorINI);
xlabel("Time (sec)");
hold on
% map extension
plot(timeLog,inlierRatio2);
% plot(timeLog,isTMM);
legend("Absolute localization error","Mapping inlier ratio","isTMM");
% %% scene change
% ylabel("Absolute trajectory error (m)");
% legend("w. TM","w.o. TM");
% saveas(1,date + "_ate_error.jpg");
disp("RMSE error: "+norm(ateError)/sqrt(idxC))
disp("max error: "+max(ateError))
disp("Loc rate: "+length(ateError)/(timePose(end)-timePose(1)))
disp("Success ratio: "+length(find(ateError < 1.0))/(timeGT(end)-timeGT(1))/10)
disp("<0.1 %: "+ length(find(ateError < 0.1))/lenPose)
disp("<0.2 %: "+ length(find(ateError < 0.2))/lenPose)
disp("<0.5 %: "+ length(find(ateError < 0.5))/lenPose)
disp("<1.0 %: "+ length(find(ateError < 1.0))/lenPose)

figure(2)
% plot(matPose(:,2),matPose(:,3),".");
plot(matPose(:,2),matPose(:,3));
hold on
plot(matGT(:,2),matGT(:,3));
% plot(matPose(idxOver1m,2),matPose(idxOver1m,3),".","MarkerSize",4);

% figure(3)
% plot(timeLog,logData{7}-logData{7}(1));
% hold on
% plot(timeLog,logData{8}-logData{8}(1));
% plot(timePose-timePose(1),matPose(:,2))
% plot(timeLog,logData{9});
% plot(timeLog,logData{10});
% legend("x","y","xF","yF");

% figure(4)
% histogram(ateError);
% hold on
% 
% figure(5)
% histogram(inlierRatio2)
% a=[timePose-timePose(1) ateError];

% figure(5)
% h = histogram(inlierRatio2, 'Normalization','probability');
% xlabel("Matching inlier ratio");
% ylabel("Probability");

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
