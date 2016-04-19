function [outcm] = comass(in)
num = size(in, 1);
inshft = [in(num-1, :)
          in
          in(2, :)];
xtotal = 0.0;
ytotal = 0.0;  
stotal = 0;
GSIZ = 64;
figrid = zeros(GSIZ, GSIZ);
   for i = 1:num-1
       bufvrt = inshft(i:i+3, :); 
       ip = [0:1:24]./25;
       incr = [-ip.*ip.*ip+3.*ip.*ip-3.*ip+1
                3.*ip.*ip.*ip-6.*ip.*ip+4
                -3.*ip.*ip.*ip+3.*ip.*ip+3.*ip+1
                ip.*ip.*ip];
       vtx = sum(repmat(bufvrt(:,1), 1, 25).*incr)./6.0;
       vty = sum(repmat(bufvrt(:,2), 1, 25).*incr)./6.0;
       figrid(GSIZ/2+fix(vty/0.05), GSIZ/2 + fix(vtx/0.05)) = 1;
   end 
   for j = 2:(GSIZ-1)
      if (figrid(1, j) == 0) & (figrid(1,j-1) == 1) & sum(figrid(1,j+1:GSIZ) > 0)
         figrid(1, j) = 1;  
     end
     if (figrid(GSIZ, j) == 0) & (figrid(GSIZ,j-1) == 1) & sum(figrid(GSIZ,j+1:GSIZ) > 0)
         figrid(GSIZ, j) = 1;  
     end
   end
   for i = 2:(GSIZ-1)
      if (figrid(i, 1) == 0) & (figrid(i-1, 1) == 1) & sum(figrid(i+1:GSIZ, 1) > 0)
         figrid(i, 1) = 1;  
     end
     if (figrid(i, GSIZ) == 0) & (figrid(i-1,GSIZ) == 1) & sum(figrid(i+1:GSIZ, GSIZ) > 0)
         figrid(i, GSIZ) = 1;  
     end
   end
 
   for i = 2:GSIZ-1
     for j = 2:GSIZ-1
     if (figrid(i, j) == 0) & (figrid(i,j-1) == 1) & (figrid(i-1,j) == 1)
       if sum(figrid(i+1:GSIZ, j) > 0)
             figrid(i, j) = 1; 
       end
     end
     end
   end

            
     stotal = sum(sum(figrid));
     xtotal = sum(sum(figrid.*repmat([1:1:GSIZ],GSIZ,1)));
     ytotal = sum(sum(figrid.*repmat([1:1:GSIZ]',1,GSIZ)));      
     
     outcm(1, 1) = (xtotal/stotal - 32)*0.05;
     outcm(1, 2) = (ytotal/stotal - 32)*0.05;

   %disp(figrid);  


