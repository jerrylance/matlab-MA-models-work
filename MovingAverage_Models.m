function MovingAverage_Models
[x,cmap]=imread('D04.hips.gif'); % Choose D04 D09 D90 D92 D24
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

W(1,1)=N*mean(mean(y(1:N,1:N)));     % Change W(0,0)
 %Parameters initialize
 %elongation alpha =[1,2,3,4,5,6,7,8]; 
 %orientation theta = [0,pi/8,2*pi/8,3*pi/8,4*pi/8,5*pi/8,6*pi/8,7*pi/8]
 Ni = 1:N;
 Nparameter=128; 
 Eta=zeros(Nparameter,N);
 Jcost=zeros(3,Nparameter); %cost function 3 parameter
 
%Cost Function
a=1;
for alpha = 1:8
    for theta = 0:pi/8:7*pi/8
        YU = gtrans(Y,alpha,theta);
        Eta(a,:) = ML_estimator(N,YU);       
        
        Jcost(2,a) = alpha;
        Jcost(3,a) = theta;
        for i=2:N/2
            if(Eta(a,i)~=0)
                Jcost(1,a)=Jcost(1,a)+(Ni(a)-1)*log(Eta(a,i))+1/2*log(Ni(a));
            end
        end
        a = a+1;
    end
end

%Minimize the cost function to obtain the alpha and theta
[~,Costmin]=min(Jcost(1));
alpha=Jcost(2,Costmin);
theta=Jcost(3,Costmin);
%After obtained the parameters, apply gtrans to Y
YU = gtrans(Y,alpha,theta);
Eta(1,:) = ML_estimator(N,YU); 
max_eta=max(Eta(1,3:N/2)); 
estimated=Eta(1,3:N/2)/max_eta;
%Synthesize
YY=zeros(N,N);
 for i=1:N,
     for m=1:N,
         for n=1:N,
         radius=hypot(m,n);
         if (radius>(i-1))&&(radius<=i),
             YY(m,n)=Eta(1,i);
         end
         end
     end
 end

YY=abs(YY).*W;
yy=ifft2(YY);
%image(abs(YY));
%pause;
colormap(gray(256));
subplot(2,2,1),imagesc(y);
title('Original texture');

c = fspecial('gaussian',3,3);% I use gaussian filter to make the result better
yy = imfilter(yy,c);
subplot(2,2,2),imagesc(real(yy));
title('Synthesized texture');

subplot(2,2,[3,4]),plot(3:N/2,estimated);
title('Estimated \eta_i');
end



            