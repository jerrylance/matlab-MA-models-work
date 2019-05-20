function Eta = ML_estimator(N,YU)
YUS = zeros(1,N);
Eta = zeros(1,N);
Ni = 1:N;
for i=1:N,
    for m=1:N
        for n=1:N
            distance=round(hypot(m,n));
            if (distance>(i-1))&&(distance<=i)
                YUS(i)=YUS(i)+(abs(YU(m,n)))^2;
            end
         end
     end
        Eta(i)=YUS(i)/Ni(i);
end
end
