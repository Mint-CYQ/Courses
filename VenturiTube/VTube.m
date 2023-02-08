d0=0.1;
k=tan(pi/9);
L=0.1;
A0=0.25*pi*d0^2;
x=linspace(0,10,50);
d=d0-2*k*x;
A=0.25*pi*d.^2;
hold on
axis([0 0.1 -0.1 0.1])
plot(x,d);
plot(x,-d);
hold off
grid on
