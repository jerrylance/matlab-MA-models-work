 function MovingAverage_residual
[x,cmap]=imread('D24.hips.gif'); % choose D17,D55,D77,D84,D68,D04,D09,D51,D68,D24
colormap(cmap);
[M,N] = size(x);
y=zeros(size(x));
for m=1:N 
    for n=1:N
        c = cmap(x(m,n)+1);
        y(m,n) = c(1);
    end
end                                         
[N,M] = size(y);
Y = fft2(y);
colormap(gray(256));
Eta=ML_estimator(N,Y);
r=Y./norm(Eta);

WI =[r(1:32,1:32).*2,0.00*randn(32,64),r(1:32,97:128).*2;0.00*randn(64,32),0.00*randn(64,64),0.00*randn(64,32);r(97:128,1:32).*2,0.00*randn(32,64),r(97:128,97:128).*2];
% residual
%imagesc(real(WI));
Wi=ifft2(WI);
%imagesc(real(Wi));

Ni=1:64;
a=1;
Jcost=zeros(3,N);
Eta=zeros(1,N);
norm_eta=zeros(N,N);
%Cost function
for alpha=1:8
    for theta=0:pi/8:7*pi/8
        YU=gtrans(Y,alpha,theta);
        Eta(a,:)=ML_estimator(N,YU); % H
        max_eta=max(Eta(a,3:N));
        norm_eta(a,3:N)=Eta(a,3:N)/(max_eta);
        Jcost(2,a)=alpha;
        Jcost(3,a)=theta;
        for i=1:128         
            if (Eta(a,i)~=0) % Eta(p,i)!=0
                Jcost(1,a)=Jcost(1,a)+(Ni(a)-1)*log(Eta(a,i))+1/2*log(Ni(a));
            end
        end        
        a=a+1;
    end
end

[~,Costmin]=min(Jcost(1));
alpha=Jcost(2,Costmin);% alpha
theta=Jcost(3,Costmin);% theta

YU=gtrans(Y,alpha,theta);
Eta(1,:)=ML_estimator(N,YU);
%synthesize
YY=zeros(N,N);
for i=1:N
    for m=1:N
        for n=1:N
            radius=hypot(m,n);
            if (radius>(i-1)&&(radius<=i))
                YY(m,n)=Eta(1,i);
            end
        end
    end
end

% White complex Gaussian process(actually use randn(N) is ok, but this way is better)
mu=randn(1,N);
sigma = N/4*eye(N);
R = chol(sigma);
w = repmat(mu,N,1) + randn(N)*R;
%construct with 0 mean (N/4)I covariance real part(same as before)
mui=randn(1,N);
sigmai = N/4*eye(N);
Ri = chol(sigmai);
wi = repmat(mui,N,1) + randn(N)*Ri;
W=w+sqrt(-1)*wi;
W1 = fft2(W); 
W = fftshift(W1);

W(1,1)=N*mean(mean(y(1:N,1:N)));   % Change W(0,0)

YY=abs(YY).*W;
yy=ifft2(YY);

subplot(3,4,1),imagesc(y);
title('Original');

subplot(3,4,2),imagesc(real(Wi));
title('Residual');


s = y.*Wi;
s = repmat(s,2,2);
subplot(3,4,[5,6,9,10]),imagesc(real(s));
title('Synthesized with residual');

yy= ifft2(Y.*W)
subplot(3,4,[7,8,11,12]),imagesc(real(yy));
title('Synthesized with white Gaussian');

