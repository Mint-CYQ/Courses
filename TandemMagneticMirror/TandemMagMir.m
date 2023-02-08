tic;clc;clear;
c0=2e-7; %c0:u0/2pi
I=1e6; %I:线圈电流(A)
La=2.0;
Lb=4.0; %L:线圈间距(m)
Ra=0.2;
Rb=0.1; %R:线圈半径(m)
dl=0.001; %dl:步长

figure
set(gcf,'position',[200 150 700 520])
subplot(121);set(gca,'position', [0.1 0.15 0.4 0.4]);
subplot(122);set(gca,'position', [0.56 0.15 0.1 0.4]);
%中心磁场
x=[];y=[];z=[]; 
r=[0.01;0.02;0.05;0.1;0.2;0.3];r=[-r;r]; %磁感线轴对称分布
for theta=0:20:180
    for j=1:length(r)
        for pm=-1:1 %磁场两边对称
            for i=1:2200    
                y(1)=r(j)*cos(theta*pi/180);
                z(1)=r(j)*sin(theta*pi/180);
                x(1)=0; %磁力线起点
                [Bx1,By1,Bz1]=magnetic0(c0,Ra,I,x(i)+La/2,y(i),z(i));%线圈1产生的磁场
                [Bx2,By2,Bz2]=magnetic0(c0,Ra,I,x(i)-La/2,y(i),z(i));%线圈2产生的磁场
                [Bx3,By3,Bz3]=magnetic0(c0,Rb,I,x(i)+Lb/2,y(i),z(i));%线圈3产生的磁场
                [Bx4,By4,Bz4]=magnetic0(c0,Rb,I,x(i)-Lb/2,y(i),z(i));%线圈4产生的磁场
                Bx=Bx1+Bx2+Bx3+Bx4;By=By1+By2+By3+By4;Bz=Bz1+Bz2+Bz3+Bz4;
                B=Bx^2+By^2+Bz^2;
                x(i+1)=x(i)+Bx/B*dl*pm;
                y(i+1)=y(i)+By/B*dl*pm;
                z(i+1)=z(i)+Bz/B*dl*pm;
            end
            subplot(121);plot3(x,y,z,'b-');hold on;box on;
            daspect([1,1.25,1.25])
            axis([-2.5,2.5,-1,1,-1,1])
            if theta==0; %截面磁场
                subplot(122);plot(y,x,'b-');hold on;
                axis([-0.5,0.5,-2,2])
            end
        end
    end
end


h=-1;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.2*sin(t),0+0.2*cos(t),'LineWidth',4) %线圈1
h=1;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.2*sin(t),0+0.2*cos(t),'LineWidth',4) %线圈2
h=-2;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.2*sin(t),0+0.1*cos(t),'LineWidth',4) %线圈3
h=2;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.2*sin(t),0+0.1*cos(t),'LineWidth',4) %线圈4

xlabel('x');ylabel('y');zlabel('z');set(gca,'FontSize',10);
subplot(122);xlabel('y');ylabel('x');set(gca,'FontSize',10);

x=-4.5:0.05:4.5;
B=c0*pi*I*Ra.^2./sqrt(Ra.^2+(La/2-x).^2)+c0*pi*I*Ra.^2./sqrt(Ra.^2+(La/2+x).^2)+c0*pi*I*Rb.^2./sqrt(Rb.^2+(Lb/2-x).^2)+c0*pi*I*Rb.^2./sqrt(Rb.^2+(Lb/2+x).^2);
figure
set(gcf,'position',[150 150 600 200])
plot(x,B)
set(gca,'position', [0.1 0.15 0.8 0.8]);
axis([-4.5,4.5,0,0.25])
xlabel('x');ylabel('B');set(gca,'FontSize',10);