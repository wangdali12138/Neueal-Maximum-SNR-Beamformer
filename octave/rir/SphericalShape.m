% generate the condinates of uniform discribed Spherical shaped microphone arrays 
function []=SphericalShape()

%the number of points
N=12; %the number of points

%According to the random surface uniform distribution, first generate an initial state
a=rand(N,1)*2*pi;
b=asin(rand(N,1)*2-1);
r0=[cos(a).*cos(b),sin(a).*cos(b),sin(b)];
v0=zeros(size(r0));


G=1e-2;%repulsion constant

%Simulation of 200 steps, generally has converged
for ii=1:200
    [rn,vn]=countnext(r0,v0,G);%update the state
    r0=rn;v0=vn;
end

plot3(rn(:,1),rn(:,2),rn(:,3),'.');hold on;%plot result
[xx,yy,zz]=sphere(50);
h2=surf(xx,yy,zz); 
set(h2,'edgecolor','none','facecolor','r','facealpha',0.7);
axis equal;
axis([-1 1 -1 1 -1 1]);
hold off;
end

%function to update state
function [rn, vn]=countnext(r,v,G) 
%rStore the x, y, z condinates of each point, v store the speed data of each point
num=size(r,1);
dd=zeros(3,num,num); %Vector difference between points
for m=1:num-1
    for n=m+1:num
        dd(:,m,n)=(r(m,:)-r(n,:))';
        dd(:,n,m)=-dd(:,m,n);
    end
end
L=sqrt(sum(dd.^2,1));%distance between points
L(L<1e-2)=1e-2; 
F=sum(dd./repmat(L.^3,[3 1 1]),3)';%Computational force

Fr=r.*repmat(dot(F,r,2),[1 3]); %Calculate the radial component of the resultant force
Fv=F-Fr; %Tangential component

rn=r+v;  %update the condinates
rn=rn./repmat(sqrt(sum(rn.^2,2)),[1 3]);
vn=v+G*Fv;%update the speed
end