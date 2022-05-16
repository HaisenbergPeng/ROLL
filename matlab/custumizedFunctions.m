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

function str = convert(R)
[n,m] = size(R);
str = "[";
for i = 1:n
    for j=1:m
        if i*j~=m*n
            str = str+num2str(R(i,j))+", ";
        else
            str = str+num2str(R(i,j))+"]";
        end
    end
end
end

function y = show(R,t)
    disp(convert(R));
    disp(convert(t));
end

function q2=rev(q)
    q2 = [q(4),q(1),q(2),q(3)];
end
