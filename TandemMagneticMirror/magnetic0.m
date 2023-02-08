%c0=u0/2pi,R:线圈半径（m），I：电流（A）
%Bx,By,Bz：磁感应强度在三个坐标上的分量
function [Bx,By,Bz]=magnetic0(c0,R,I,x,y,z)
rho=sqrt(z^2+y^2);
theta=atan2(z,y);      %tan(y/x)=theta,theta为幅度值   
k2=4*R*rho/((R+rho)^2+x^2);	%椭圆函数中的参数k方
k2=k2-eps;                  
[K,E]=ellipke(k2);
dd=(R-rho)^2+x^2;
dd=dd+eps; %防止分母为零
B_rho=c0*I*x/((rho+1e-3)*sqrt((R+rho)^2+x^2))*((R^2+rho^2+x^2)/dd*E-K);
B_x=c0*I/sqrt((R+rho)^2+x^2)*((R^2-rho^2-x^2)/dd*E+K);
B_theta=0;
By=B_rho*cos(theta)-B_theta*sin(theta);
Bz=B_rho*sin(theta)+B_theta*cos(theta);
Bx=B_x;
end