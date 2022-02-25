clc;
clear
close all
figure(3)

box on
set(gca,'XTick',[]);
set(gca,'XLim',[0 27]);
set(gca,'YLim',[0 100]);

color_matrix = [1,0,0                
                0,1,0         
                0,0,1]; 
X1= [10,20,70;
    20,30,50;
    30,40,30;
    10,40,50];
X2= [10,20,70;
    20,30,50;
    30,40,30;
    10,40,50];
hold on
for i = 1:4
    b = bar(i+1:i+3,[X1(i,:);0,0,0;0,0,0],'stacked');
    set(b(1),'facecolor',color_matrix(1,:));
    set(b(2),'facecolor',color_matrix(2,:));
    set(b(3),'facecolor',color_matrix(3,:));
    
    b = bar(i+6:i+8,[X2(i,:);0,0,0;0,0,0],'stacked');
    set(b(1),'facecolor',color_matrix(1,:));
    set(b(2),'facecolor',color_matrix(2,:));
    set(b(3),'facecolor',color_matrix(3,:));
end

ylabel('Percentage (%)') 
set(gca,'FontSize',15,'Fontname', 'Arial');
