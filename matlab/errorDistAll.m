clc;
clear;
close all
startDate = "2012-01-15";
dateLists = ["2012-02-02","2012-03-17","2012-04-29",...
    "2012-05-11","2012-06-15","2012-08-04","2012-11-17","2013-01-10","2013-02-23"]';
setNum = length(dateLists);
daysPassed = cntDays(startDate,dateLists);
folder = "/media/haisenberg/BIGLUCK/Datasets/NCLT/datasets/fastlio_mapping";
ateErrorCell = cell(setNum,1);
percWithinOneMeter = zeros(setNum,1);
stdError = zeros(setNum,1);
meanError = zeros(setNum,1);
maxError = zeros(setNum,1);
TMMno = zeros(setNum,1);
figure(1)
for iB=1:setNum
    date = dateLists{iB};
    disp("reading "+date);
    logFilePath = folder+"/"+date+"/map_pcd/mappingError.txt";
    poseFilePath = folder+"/"+date+"/map_pcd/path_mapping.txt";
%     poseFilePath = folder+"/"+date+"/map_pcd/path_fusion.txt"; 
    gtFilePath = "/media/haisenberg/BIGLUCK/Datasets/NCLT/datasets/"+date+"/groundtruth_"+date+".csv";

    %% log file reading
    fID = fopen(logFilePath);
    logData = textscan(fID,strPatternGenerate(11));
    timeLog = logData{1}-logData{1}(1);
    regiError = logData{5};
    inlierRatio2 = logData{4};
    inlierRatio = logData{3};
    isTMM = logData{2};

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
    matGT = zeros(floor(lenGT/10),7);
    %% kloam+fastlio uses imu pose, so here convert body pose to imu pose
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
    %% sync with time
    timeGT = matGT(:,1)/1e+6; % us -> sec
    timePose =  matPose(:,1)/1e+6;
    MDtimeGT = KDTreeSearcher(timeGT);
    [idx, D] = rangesearch(MDtimeGT,timePose,0.05);
    ateError = zeros(lenPose,1);
    not_found = 0;

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
        
        %% convert gt body to gt imu
        ateError(i) = norm(matPose(i,2:3)-matGT(idx{i}(1),2:3));
    end
    ateErrorCell{iB} = ateError;
    percWithinOneMeter(iB) = 100 -length(find(ateError> 1.0))/length(ateError)*100;
    stdError(iB) = std(ateError);
    meanError(iB) = mean(ateError);
    maxError(iB)= max(ateError);
    TMMno(iB) = TMMcnt(isTMM);
    
    histogram(ateError,"DisplayStyle","stairs");
    hold on
end
legend(dateLists);
hold off
xlabel("Localization error (m)");
ylabel("Count");
%% PLOT and SAVE
T = table(daysPassed,meanError,stdError,maxError,percWithinOneMeter,TMMno);
writetable(T);

% a=[timePose-timePose(1) ateError];
function strPattern =strPatternGenerate(n)
    strPattern = "";
    for i=1:n
        strPattern = strPattern+"%f";
    end

end

function n = TMMcnt(x)
    n=0;
    for i=1:length(x)-1
        if x(i+1) -x(i) > 0.9
            n = n+1;
        end
    end
end

function days = cntDays(start,str)
    n = length(str);
    days = zeros(n,1);
    for i=1:n
        if i==1
            day1 =convertNum(start) ;
        else
            day1 = convertNum(str(i-1));
        end
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