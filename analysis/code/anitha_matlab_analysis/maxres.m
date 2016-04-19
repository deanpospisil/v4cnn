function [df] = valres(start)
global resp fe err dim 

count = 1;

for i = 1:370
    t = count + resp(i,1) - 1;%the index up to which has these parameters.
    vec = fe(count:t, :);%this particular shapes orxcurv params
    den = repmat(start(2,1:dim), resp(i,1), 1);
    zee = (vec - repmat(start(1,1:dim),resp(i,1),1));%take the shape params and subtract off, the starting point
    % making ang. position circular for the first dimension
    t1 = abs(vec(:,1) - start(1,1));%absolute difference in 
    t2 = 2*pi-t1;
    
    zee(:,1) = le(t1,t2).*t1+gt(t1,t2).*t2;
    zee = zee.*den;
    df(i) = resp(i,2) - max(1./(exp(sum(zee.*zee,2)./2.0)))*start(1, dim+1);
    err(i, :) = [resp(i,2) resp(i,2)-df(i)];
    count = count + resp(i,1);
%     
%     veclist{i}=vec;
%     save('StimAngCurCoords','veclist')

end

