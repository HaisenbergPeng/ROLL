% clc
clear;
close all
folder = "/media/haisenberg/500C-188D/实验/";
poseFilePath = folder+"FrameTrajectory_TUM_Format.txt";
gtFilePath =folder + "path_6DOF_frame.txt";

%% pose file reading
fID2 = fopen(poseFilePath);
strPattern = "";
n = 8;
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
%% gt file reading
fID2 = fopen(gtFilePath);
strPattern = "";
n = 8;
for i=1:n
    strPattern = strPattern+"%f";
end
poseData = textscan(fID2,strPattern);
lenPose = length(poseData{1});
gtPose = zeros(lenPose,7);
for i=1:lenPose
    for j=1:7
        gtPose(i,j) = poseData{j}(i);
    end
end

plot(matPose(:,2),matPose(:,3));
hold on
plot(gtPose(:,3),gtPose(:,4));
legend("pose","gt")

%% sync with time
timeGT = matGT(:,1); % us -> sec
timePose =  matPose(:,1);
[idx, D] = rangesearch(timeGT,timePose,0.05);
yawError = zeros(lenPose,1);

not_found = 0;
idxC = 0;
for i=1:lenPose
    if isempty(idx{i})
        not_found = not_found + 1;
        continue;    
    end
    idxC = idxC + 1;
    deltaT = transError(matGT(idx{i}(1),2:7),matPose(i,2:7));
    ateErrorINI(idxC) = norm(deltaT(1:3,4));
%     if ateErrorINI(idxC) > 10
%         i
%         matPose(i,1)-matGT(idx{i}(1),1)
%         timePose(i)-timeGT(idx{i}(1))
%         matPose(i,2:7)
%     end
    %% 2pi
%     Ntmp = round(yawError(i)/360);
%     yawError(i) = yawError(i) - Ntmp*360;
%     if ateError(i)>1
%         matGT(idx{i}(1),2:3)
%     end
end
ateError = ateErrorINI(1:idxC);
idxOver1m= find(ateError > 1.0);
% %% PLOT
% figure(1)
% plot(timePose-timePose(1),ateErrorINI);
% xlabel("Time (sec)");
% hold on
% plot(timeLog,inlierRatio2);
% plot(timeLog,isTMM);
% legend("Absolute localization error","Mapping inlier ratio","isTMM");

disp("RMSE error: "+norm(ateError)/sqrt(idxC))
disp("max error: "+max(ateError))
disp("Loc rate: "+length(ateError)/(timePose(end)-timePose(1)))
disp("Success ratio: "+length(find(ateError < 2.0))/(timeGT(end)-timeGT(1))/10)
disp("<0.1 %: "+ length(find(ateError < 0.1))/lenPose)
disp("<0.2 %: "+ length(find(ateError < 0.2))/lenPose)
disp("<0.5 %: "+ length(find(ateError < 0.5))/lenPose)
disp("<1.0 %: "+ length(find(ateError < 1.0))/lenPose)

figure(1)
% plot(matPose(:,2),matPose(:,3),".");
plot(matPose(:,2),matPose(:,3));
hold on
plot(matGT(:,2),matGT(:,3));
% plot(matPose(idxOver1m,2),matPose(idxOver1m,3),".","MarkerSize",4);

figure(2)
subplot(3,1,1)
plot(timePose-timePose(1),);
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
    T1 = eye(4);
    T2 = eye(4);
    T1(1:3,1:3) = eul2rotm([Vgt(6),Vgt(5),Vgt(4)],"ZYX");
    T2(1:3,1:3) = eul2rotm([V2(6),V2(5),V2(4)],"ZYX");
    T1(1:3,4) = Vgt(1:3)';
    T2(1:3,4) = V2(1:3)';
    eT = T1\T2;    % equals to T1\T2
end
