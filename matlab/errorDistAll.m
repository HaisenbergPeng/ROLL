% clc;
clear;
close all
startDate = "2012-01-15";
% %% LOMAL
% dateLists = ["2012-02-02","2012-03-17","2012-04-29",...
%     "2012-05-11","2012-06-15","2012-08-04","2012-11-17","2013-01-10","2013-02-23"]';
% folder = "/mnt/sdb/Datasets/NCLT/datasets/logs/roll";

% %% no TM
% dateLists = ["2012-02-02","2012-03-17","2012-04-29","2012-05-11"]';
% folder = "/mnt/sdb/Datasets/NCLT/datasets/logs/fastlio_noTMM";

% %% no CC
% dateLists = ["2012-02-02","2012-03-17","2012-04-29","2012-05-11"]';
% folder = "/mnt/sdb/Datasets/NCLT/datasets/logs/fastlio_noCC";

%% no edge
dateLists = ["2012-02-02","2012-03-17","2012-04-29","2012-05-11"]';
folder = "/mnt/sdb/Datasets/NCLT/datasets/logs/fastlio_noEdge";

setNum = length(dateLists);
daysPassed = cntDays(startDate,dateLists);

ateErrorCell = cell(setNum,1);

percWithinOneMeter = zeros(setNum,1);
percWithin05Meter = zeros(setNum,1);
percWithin02Meter = zeros(setNum,1);
percWithin01Meter = zeros(setNum,1);
stdError = zeros(setNum,1);
meanError = zeros(setNum,1);
maxError = zeros(setNum,1);

RMSE = zeros(setNum,1);
maxTUM = zeros(setNum,1);

locFrequency =  zeros(setNum,1);
locFrequencyG =  zeros(setNum,1);
successRate =  zeros(setNum,1);
TMMno = zeros(setNum,1);
duration = zeros(setNum,1);
mapExtension = cell(setNum,1);
timeAll = 0;
for iB=1:setNum
    date = dateLists{iB};
    disp("reading "+date);
    logFilePath = folder+"/"+date+"/map_pcd/mappingError.txt";
%     poseFilePath = folder+"/"+date+"/map_pcd/path_mapping.txt";
%     poseFilePath = folder+"/"+date+"/map_pcd/path_fusion.txt"; 
    poseFilePath = folder+"/"+date+"/map_pcd/path_vinsfusion.txt";
    gtFilePath = "/mnt/sdb/Datasets/NCLT/datasets/ground_truth"+"/groundtruth_"+date+".csv";

    %% log file reading
    fID = fopen(logFilePath);
    logData = textscan(fID,strPatternGenerate(11));
    timeLog = logData{1}-logData{1}(1);
    regiError = logData{5};
    inlierRatio2 = logData{4};
    inlierRatio = logData{3};
    isTMM = logData{2};
    
    if any(inlierRatio2 <0.1)
        mapExtension{iB} = "YES";
    else
        mapExtension{iB} = "NO";
    end
    %% pose file reading
    fID2 = fopen(poseFilePath);
    poseData = textscan(fID2,strPatternGenerate(7));
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
    MDtimeGT = KDTreeSearcher(timeGT);
    [idx, D] = rangesearch(MDtimeGT,timePose,0.05);
    ateErrorInit = zeros(lenPose,1);% < 1% mismatch
    ateTUMinit = zeros(lenPose,1);
    not_found = 0;

    idxC = 0;
    for i=1:lenPose
        if isempty(idx{i})
            not_found = not_found + 1;
            continue;    
        end
        idxC = idxC + 1;
        %% rule out obvious wrong ground truth, please check for yourself
        if date=="2013-02-23" && matPose(i,2)>-310 && matPose(i,2)<-260&&...
            matPose(i,3)>-450 && matPose(i,3)<-435
            continue;
        end
        ateErrorInit(idxC) = norm(matPose(i,2:3)-matGT(idx{i}(1),2:3));
        deltaT = transError(matGT(idx{i}(1),2:7),matPose(i,2:7));
        ateTUMinit(idxC) = norm(deltaT(1:3,4));
    end
    ateError = ateErrorInit(1:idxC);
    ateTUM = ateTUMinit(1:idxC);
%     disp("Not found: "+ num2str(not_found/lenPose)); % < 1%
    duration(iB) = (timeGT(end)-timeGT(1))/3600; 
    [TMMno(iB), positions] = TMMcnt(isTMM);
    locFrequency(iB) = lenPose/(timeGT(end)-timeGT(1));
    locFrequencyG(iB) = length(timeLog)/(timeGT(end)-timeGT(1));
    %% TUM
    RMSE(iB) = norm(ateTUM)/sqrt(idxC);

    ateErrorCell{iB} = ateTUM;
    percWithinOneMeter(iB) = length(find(ateTUM < 1.0))/length(ateTUM)*100;
    percWithin05Meter(iB) = length(find(ateTUM < 0.5))/length(ateTUM)*100;
    percWithin02Meter(iB) = length(find(ateTUM < 0.2))/length(ateTUM)*100;
    percWithin01Meter(iB) = length(find(ateTUM < 0.1))/length(ateTUM)*100;
    stdError(iB) = std(ateTUM);
    meanError(iB) = mean(ateTUM);
    maxError(iB)= max(ateTUM);
    successRate(iB) = length(find(ateTUM < 1.0))/10/(timeGT(end)-timeGT(1))*100;
    timeAll = timeAll + timeGT(end)-timeGT(1);
    figure(1)
    histogram(ateTUM,"DisplayStyle","stairs");
    hold on
%     
%     figure(2)
%     idxOver1m = find(ateError>1.0);
%     plot(matPose(idxOver1m,2),matPose(idxOver1m,3),".","MarkerSize",10);
%     hold on
    
    figure(3)
    y = matPose(:,3);
    y(end) = nan;
    patch(matPose(:,2),y,ateTUMinit,'EdgeColor','interp','MarkerFaceColor','flat');
    hold on
    
    for iT = 1:TMMno(iB)
        plot(matPose(positions(iT),2),matPose(positions(iT),3),'o','MarkerFaceColor','r');
    end
    
end
colorbar;
hold off

figure(1)
legend(dateLists);
xlabel("Localization error (m)");
ylabel("Count");

figure(3)
xlabel('X (m)');
ylabel('Y (m)');
%% PLOT and SAVE
T = table(dateLists,daysPassed,duration,RMSE,meanError,maxError,percWithin01Meter,percWithin02Meter,...
    percWithin05Meter,percWithinOneMeter,successRate, locFrequency,TMMno,mapExtension);
writetable(T,"results/loc_results_noEdge.xls");

%% overall mean and std
ateAll = [];
for i=1:setNum
    ateAll = [ateAll; ateErrorCell{i}];
end
disp("--------------ROLL----------------");
disp("RMSE error for all: "+ norm(ateAll)/sqrt(length(ateAll)));
disp("MAX error for all: " + max(ateAll));
disp("0.1 m percent for all: " + 100*length(find(ateAll<0.1))/length(ateAll) );
disp("0.2 m percent for all: " + 100*length(find(ateAll<0.2))/length(ateAll));
disp("0.5 m percent for all: " + 100*length(find(ateAll<0.5))/length(ateAll)  );
disp("1.0 m percent for all: " + 100*length(find(ateAll<1.0))/length(ateAll)  );
disp("Loc rate: "+length(ateAll)/timeAll)
% disp("Success ratio: "+length(find(ateError < 1.0))/(timeGT(end)-timeGT(1))/10)
%% error space distribution
gtFilePath = "/mnt/sdb/Datasets/NCLT/datasets/"+startDate+"/groundtruth_"+startDate+".csv";
fID3 = fopen(gtFilePath);
gtData = textscan(fID3, "%f%s%f%s%f%s%f%s%f%s%f%s%f");
% downsample
downsample = 10;
lenGT = length(gtData{1});
matGT = zeros(floor(lenGT/10),7);
tbi = [-0.11 -0.18 -0.71]';
for i=1:floor(lenGT/10)
    for j=1:7
        matGT(i,j) = gtData{2*j-1}(10*i);
    end
    Rmb = eul2rotm([ matGT(i,7),matGT(i,6),matGT(i,5)],"ZYX");
    tmb = matGT(i,2:4)';
    tmi = Rmb*tbi +tmb;
    matGT(i,2:4) = tmi';
end
% figure(4)
% histogram(ateAll);

% a=[timePose-timePose(1) ateError];
function strPattern =strPatternGenerate(n)
    strPattern = "";
    for i=1:n
        strPattern = strPattern+"%f";
    end

end

function [n, positions] = TMMcnt(x)
    n=0;
    positions = zeros(10,1);
    for i=1:length(x)-1
        if x(i+1) -x(i) > 0.9
            n = n+1;
            positions(n) = i;
        end
    end
end

function days = cntDays(start,str)
    n = length(str);
    days = zeros(n,1);
    day1 =convertNum(start) ;
    for i=1:n
        day2 = convertNum(str(i));
        days(i) = datenum(day2(1),day2(2),day2(3))-datenum(day1(1),day1(2),day1(3));        
    end
end
function nums = convertNum(x)
    strCell = split(x,"-");
    nums = zeros(3,1);
    for i=1:3
        nums(i) = strCell(i);
    end
end

function eT = transError(Vgt,V2)
% input: x y z r p y
    T1 = eye(4);
    T2 = eye(4);
    T1(1:3,1:3) = eul2rotm([Vgt(6),Vgt(5),Vgt(4)],"ZYX");
    T2(1:3,1:3) = eul2rotm([V2(6),V2(5),V2(4)],"ZYX");
    T1(1:3,4) = Vgt(1:3);
    T2(1:3,4) = V2(1:3);
    eT = inv(T1)*T2;    
end
