clear;
clc;
g=1.4;%绝热指数
Nt=801;%时间步长
Nx=21;%空间网格数目
L=0.1;%喷管长度
C=0.5;%柯朗数
k=tan(pi/9);
d_0=0.1;%喷管最左端直径
A_0=0.25*pi*d_0^2;
x_x=linspace(0,L,Nx);
d_x=d_0-2*k*x_x;
A_x=0.25*pi*d_x.^2;%尺寸

x=linspace(0,1,Nx);%网格点坐标
d=d_x./d_0;
A=A_x./A_0;%无量纲化
dx=L/(Nx-1); %空间步长
dt(Nt)=0;%时间步长
%x=linspace(0,L,Nx);%网格点横坐标
r(Nt,Nx)=0;
T(Nt,Nx)=0; 
v(Nt,Nx)=0;%初始化
r(1,:)=1-0.03146*x;
T(1,:)=1-0.02314*x;
v(1,:)=(0.1+1.09*x).*(1-0.02314*x).^0.5;%初值条件

for k=1:Nt-1%各k时刻计算i点值
    %预估步计算偏导数（向前差分）
    r_t(1:Nx-1)=-v(k,1:Nx-1).*(r(k,2:Nx)-r(k,1:Nx-1))./dx-r(k,1:Nx-1).*(v(k,2:Nx)-v(k,1:Nx-1))/dx-r(k,1:Nx-1).*v(k,1:Nx-1).*log(A(2:Nx)./A(1:Nx-1))/dx;
    v_t(1:Nx-1)=-v(k,1:Nx-1).*(v(k,2:Nx)-v(k,1:Nx-1))./dx-1/g.*((T(k,2:Nx)-T(k,1:Nx-1))/dx+T(k,1:Nx-1)./r(k,1:Nx-1).*(r(k,2:Nx)-r(k,1:Nx-1))./dx);
    T_t(1:Nx-1)=-v(k,1:Nx-1).*(T(k,2:Nx)-T(k,1:Nx-1))./dx-(g-1).*T(k,1:Nx-1).*((v(k,2:Nx)-v(k,1:Nx-1))/dx+v(k,1:Nx-1).*log(A(2:Nx)./A(1:Nx-1))/dx);
    %求取内部网格点处最小时间步长
    t=C*dx./(v(k, 2:Nx-1)+sqrt(T(k, 2:Nx-1)));
    dt(k)=min(t);
    %预估步计算预估值
    r1(1:Nx-1)=r(k,1:Nx-1)+r_t(1:Nx-1)*dt(k);
    v1(1:Nx-1)=v(k,1:Nx-1)+v_t(1:Nx-1)*dt(k);
    T1(1:Nx-1)=T(k,1:Nx-1)+T_t(1:Nx-1)*dt(k);
    %校正步计算偏导（向后差分）
    r_t_1(2:Nx-1)=-v1(2:Nx-1).*(r1(2:Nx-1)-r1(1:Nx-2))./dx-r1(2:Nx-1).*(v1(2:Nx-1)-v1(1:Nx-2))./dx-r1(2:Nx-1).*v1(2:Nx-1).*log(A(2:Nx-1)./A(1:Nx-2))./dx;
    v_t_1(2:Nx-1)=-v1(2:Nx-1).*(v1(2:Nx-1)-v1(1:Nx-2))./dx-1/g.*(T1(2:Nx-1)-T1(1:Nx-2)./dx+T1(2:Nx-1)./r1(2:Nx-1).*(r1(2:Nx-1)-r1(1:Nx-2))./dx);
    T_t_1(2:Nx-1)=-v1(2:Nx-1).*(T1(2:Nx-1)-T1(1:Nx-2))./dx-(g-1).*T1(2:Nx-1).*((v1(2:Nx-1)-v1(1:Nx-2))./dx+v1(2:Nx-1).*log(A(2:Nx-1)./A(1:Nx-2))./dx);
    %偏导数平均值
    r_t_av(2:Nx-1)=0.5*(r_t(2:Nx-1)+r_t_1(2:Nx-1));
    v_t_av(2:Nx-1)=0.5*(v_t(2:Nx-1)+v_t_1(2:Nx-1)); 
    T_t_av(2:Nx-1)=0.5*(T_t(2:Nx-1)+T_t_1(2:Nx-1)); 
    %校正步计算校正值（内部点）
    r(k+1,2:Nx-1)=r(k,2:Nx-1)+r_t_av(2:Nx-1)*dt(k); 
    v(k+1,2:Nx-1)=v(k,2:Nx-1)+v_t_av(2:Nx-1)*dt(k);
    T(k+1,2:Nx-1)=T(k,2:Nx-1)+T_t_av(2:Nx-1)*dt(k);
    %出口边界值
    r(k+1,Nx)=2*r(k+1,Nx-1)-r(k+1,Nx-2); 
    v(k+1,Nx)=2*v(k+1,Nx-1)-v(k+1,Nx-2);
    T(k+1,Nx)=2*T(k+1,Nx-1)-T(k+1,Nx-2);
    %入口边界值
    r(k+1,1)=1;
    v(k+1,1)=2*v(k+1,2)-v(k+1,3);
    T(k+1,1)=1;
end

p(Nt,Nx)=1;
for k=1:Nt
    p(k,1:Nx)=1-r(k,1:Nx).*(v(k,1:Nx)).^2;
end
vx(1:Nx)=v(Nt,1:Nx);
px(1:Nx)=p(Nt,1:Nx);
rx(1:Nx)=r(Nt,1:Nx);
C(1:Nx)=vx(1:Nx).*rx(1:Nx).*A(1:Nx);
for k=1:Nt
    C1(k,1:Nx)=v(k,1:Nx).*r(k,1:Nx).*A(1:Nx);
end
hold on
axis([0 1 0 4.8])
%title('质量流量')
plot(x,C1);
hold off
xlabel('x')
ylabel('Q')
grid on

%{
title('v最终数值解')
plot(x,vx);
xlabel('x')
ylabel('v')
grid on

figure
title('p最终数值解')
plot(x,px);
xlabel('x')
ylabel('p')
grid on

title('v数值解')
plot(x,v);
xlabel('x')
ylabel('v')
grid on

figure
title('p数值解')
plot(x,p);
xlabel('x')
ylabel('p')
grid on

tt(1:Nt)=0;
tt(1)=dt(1);
tt(2:Nt)=tt(1:Nt-1)+dt(2:Nt);
t=linspace(tt(Nt),tt(1),Nt);
%figure
plot(t,v(:,11))

xt(1:Nx-1)=x(1:Nx-1);
pvpt(1:Nt-1,1:Nx)=v(2:Nt,:)-v(1:Nt-1,:);
pppt(1:Nt-1,1:Nx)=p(2:Nt,:)-p(1:Nt-1,:);

title('v偏导数')
plot(x,pvpt);
xlabel('x')
ylabel('v')
grid on

figure
title('p偏导数')
plot(x,pppt);
xlabel('x')
ylabel('p')
grid on
%}