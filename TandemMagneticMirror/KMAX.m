tic;clc;clear;
c0=2e-7; %c0:u0/2pi
Ia1=5e2;
Ia2=2e3;
Ia3=3e3;
Ib1=2e3;
Ib2=3e3;
Ib3=20;
Ih=5e4;%I:线圈电流(A)
La1=0.44;
La2=1.32;
La3=2.40;
Lb1=4.30;
Lb2=7.20;
Lb3=10.90;
Lh1=5.22;
Lh2=9.22;%L:线圈间距(m)
Ra=0.92;
Rb=0.81; 
Rh=0.23;%R:线圈半径(m)
dl=0.001; %dl:步长

figure
set(gcf,'position',[200 150 700 520])
subplot(121);set(gca,'position', [0.1 0.15 0.4 0.4]);
subplot(122);set(gca,'position', [0.56 0.15 0.1 0.4]);
x=[];y=[];z=[]; 
r=[0.01;0.02;0.05;0.1;0.15];r=[-r;r]; %磁感线轴对称分布
for theta=0:30:180
    for j=1:length(r)
        for pm=-1:1 %磁场两边对称
            for i=1:140   
                y(1)=r(j)*cos(theta*pi/180);
                z(1)=r(j)*sin(theta*pi/180);
                x(1)=0; %磁力线起点
                [Bxa11,Bya11,Bza11]=magnetic0(c0,Ra,Ia1,x(i)+La1/2,y(i),z(i));%线圈对A1产生的磁场
                [Bxa12,Bya12,Bza12]=magnetic0(c0,Ra,Ia1,x(i)-La1/2,y(i),z(i));%线圈对A1产生的磁场
                [Bxa21,Bya21,Bza21]=magnetic0(c0,Ra,Ia2,x(i)+La2/2,y(i),z(i));%线圈对A2产生的磁场
                [Bxa22,Bya22,Bza22]=magnetic0(c0,Ra,Ia2,x(i)-La2/2,y(i),z(i));%线圈对A2产生的磁场
                [Bxa31,Bya31,Bza31]=magnetic0(c0,Ra,Ia3,x(i)+La3/2,y(i),z(i));%线圈对A3产生的磁场
                [Bxa32,Bya32,Bza32]=magnetic0(c0,Ra,Ia3,x(i)-La3/2,y(i),z(i));%线圈对A3产生的磁场
                [Bxb11,Byb11,Bzb11]=magnetic0(c0,Rb,Ib1,x(i)+Lb1/2,y(i),z(i));%线圈对B1产生的磁场
                [Bxb12,Byb12,Bzb12]=magnetic0(c0,Rb,Ib1,x(i)-Lb1/2,y(i),z(i));%线圈对B1产生的磁场
                [Bxb21,Byb21,Bzb21]=magnetic0(c0,Rb,Ib2,x(i)+Lb2/2,y(i),z(i));%线圈对B2产生的磁场
                [Bxb22,Byb22,Bzb22]=magnetic0(c0,Rb,Ib2,x(i)-Lb2/2,y(i),z(i));%线圈对B2产生的磁场
                [Bxb31,Byb31,Bzb31]=magnetic0(c0,Rb,Ib3,x(i)+Lb3/2,y(i),z(i));%线圈对B3产生的磁场
                [Bxb32,Byb32,Bzb32]=magnetic0(c0,Rb,Ib3,x(i)-Lb3/2,y(i),z(i));%线圈对B3产生的磁场
                [Bxh11,Byh11,Bzh11]=magnetic0(c0,Rh,Ih,x(i)+Lh1/2,y(i),z(i));%线圈对H1产生的磁场
                [Bxh12,Byh12,Bzh12]=magnetic0(c0,Rh,Ih,x(i)-Lh1/2,y(i),z(i));%线圈对H1产生的磁场
                [Bxh21,Byh21,Bzh21]=magnetic0(c0,Rh,Ih,x(i)+Lh2/2,y(i),z(i));%线圈对H2产生的磁场
                [Bxh22,Byh22,Bzh22]=magnetic0(c0,Rh,Ih,x(i)-Lh2/2,y(i),z(i));%线圈对H2产生的磁场
                Bx=Bxa11+Bxa12+Bxa21+Bxa22+Bxa31+Bxa32+Bxb11+Bxb12+Bxb21+Bxb22+Bxb31+Bxb32+Bxh11+Bxh12+Bxh21+Bxh22;
                By=Bya11+Bya12+Bya21+Bya22+Bya31+Bya32+Byb11+Byb12+Byb21+Byb22+Byb31+Byb32+Byh11+Byh12+Byh21+Byh22;
                Bz=Bza11+Bza12+Bza21+Bza22+Bza31+Bza32+Bzb11+Bzb12+Bzb21+Bzb22+Bzb31+Bzb32+Bzh11+Bzh12+Bzh21+Bzh22;
                B=Bx^2+By^2+Bz^2;
                x(i+1)=x(i)+Bx/B*dl*pm;
                y(i+1)=y(i)+By/B*dl*pm;
                z(i+1)=z(i)+Bz/B*dl*pm;
            end
            subplot(121);plot3(x,y,z,'b-');hold on;box on;
            daspect([1,1.35,1.35])
            axis([-5.5,5.5,-1,1,-1,1])
            if theta==0; %截面磁场
                subplot(122);plot(y,x,'b-');hold on;
                axis([-0.5,0.5,-5.5,5.5])
            end
        end
    end
end


h=-0.22;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.41*sin(t),0+0.41*cos(t),'LineWidth',4) %线圈对A1
h=0.22;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.41*sin(t),0+0.41*cos(t),'LineWidth',4) %线圈对A1
h=-0.66;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.41*sin(t),0+0.41*cos(t),'LineWidth',4) %线圈对A2
h=0.66;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.41*sin(t),0+0.41*cos(t),'LineWidth',4) %线圈对A2
h=-1.20;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.41*sin(t),0+0.41*cos(t),'LineWidth',4) %线圈对A3
h=1.20;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.41*sin(t),0+0.41*cos(t),'LineWidth',4) %线圈对A3
h=-2.15;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.37*sin(t),0+0.37*cos(t),'LineWidth',4) %线圈对B1
h=2.15;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.37*sin(t),0+0.37*cos(t),'LineWidth',4) %线圈对B1
h=-3.6;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.37*sin(t),0+0.37*cos(t),'LineWidth',4) %线圈对B2
h=3.6;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.37*sin(t),0+0.37*cos(t),'LineWidth',4) %线圈对B2
h=-5.4;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.37*sin(t),0+0.37*cos(t),'LineWidth',4) %线圈对B3
h=5.4;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.37*sin(t),0+0.37*cos(t),'LineWidth',4) %线圈对B3
h=-2.61;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.2*sin(t),0+0.1*cos(t),'LineWidth',4) %线圈对H1
h=2.61;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.2*sin(t),0+0.1*cos(t),'LineWidth',4) %线圈对H1
h=-4.61;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.2*sin(t),0+0.1*cos(t),'LineWidth',4) %线圈对H2
h=4.61;t=0:0.1:(2*pi);t=[t,0];
plot3(h*ones(size(t)),0+0.2*sin(t),0+0.1*cos(t),'LineWidth',4) %线圈对H2


xlabel('x');ylabel('y');zlabel('z');set(gca,'FontSize',10);
subplot(122);xlabel('y');ylabel('x');set(gca,'FontSize',10);

x=-8:0.05:8;
B=c0*pi*Ia1*(Ra.^2./sqrt(Ra.^2+(La1/2-x).^2)+Ra.^2./sqrt(Ra.^2+(La1/2+x).^2)+ ...
    Ra.^2./sqrt(Ra.^2+(La2/2-x).^2)+Ra.^2./sqrt(Ra.^2+(La2/2+x).^2)+ ...
    Ra.^2./sqrt(Ra.^2+(La3/2-x).^2)+Ra.^2./sqrt(Ra.^2+(La3/2+x).^2))+ ...
    c0*pi*Ib1*(Rb.^2./sqrt(Rb.^2+(Lb1/2-x).^2)+Rb.^2./sqrt(Rb.^2+(Lb1/2+x).^2)+ ...
    Rb.^2./sqrt(Rb.^2+(Lb2/2-x).^2)+Rb.^2./sqrt(Rb.^2+(Lb2/2+x).^2)+ ...
    Rb.^2./sqrt(Rb.^2+(Lb3/2-x).^2)+Rb.^2./sqrt(Rb.^2+(Lb3/2+x).^2))+ ...
    c0*pi*Ih*(Rh.^2./sqrt(Rh.^2+(Lh1/2-x).^2)+Rh.^2./sqrt(Rh.^2+(Lh1/2+x).^2)+ ...
    Rh.^2./sqrt(Rh.^2+(Lh2/2-x).^2)+Rh.^2./sqrt(Rh.^2+(Lh2/2+x).^2));
figure
set(gcf,'position',[150 150 600 200])
plot(x,B)
set(gca,'position', [0.1 0.15 0.8 0.8]);
axis([-8,8,0,0.016])
xlabel('x');ylabel('B');set(gca,'FontSize',10);
