function [outvec] = fvmax(invec)
  sample = 100.0;
  num = size(invec, 1);
  inshft = [invec(num-1, :)
          invec
          invec(2, :)];
      ip = [0:1:49]./50;
      
  for i = 1:num-1
    bufvrt = inshft(i:i+3, :); 
    incr = [-ip.*ip.*ip+3.*ip.*ip-3.*ip+1
             3.*ip.*ip.*ip-6.*ip.*ip+4
            -3.*ip.*ip.*ip+3.*ip.*ip+3.*ip+1
             ip.*ip.*ip];
         
    dincr = [-3.*ip.*ip+6.*ip-3
             9.*ip.*ip-12.*ip
            -9.*ip.*ip+6.*ip+3
             3.*ip.*ip];
         
    vtx(1, i*50-49:i*50) = sum(repmat(bufvrt(:,1), 1, 50).*incr)./6.0;
    vty(1, i*50-49:i*50) = sum(repmat(bufvrt(:,2), 1, 50).*incr)./6.0;
    dvtx(1, i*50-49:i*50) = sum(repmat(bufvrt(:,1), 1, 50).*dincr)./6.0;
    dvty(1, i*50-49:i*50) = sum(repmat(bufvrt(:,2), 1, 50).*dincr)./6.0;
  end 
  vtx(1, num*50-49) = vtx(1, 1);
  vty(1, num*50-49) = vty(1, 1);
  outvec = [vtx' vty'];    