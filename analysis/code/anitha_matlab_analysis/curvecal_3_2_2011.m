function [curv, angpos] = curvecal(obnum, invec, com)
  sample = 50.0;
  if (obnum == 1) | (obnum == 2)
    numrot = 1;
  elseif (obnum == 32) | (obnum == 37)
    numrot = 2;
  elseif (obnum == 5) | (obnum == 7) | (obnum == 34)
    numrot = 4;
  else 
    numrot = 8;
  end
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
  dincr50 = [0 -3 0 3]';
  dvtx(1, num*50-49) = sum(bufvrt(:,1).*dincr50)./6.0;
  dvty(1, num*50-49) = sum(bufvrt(:,2).*dincr50)./6.0;
  
  tgtang = atan2(dvty, dvtx) + lt(atan2(dvty, dvtx), 0.0).*2.*pi;
  dtheta = diff(tgtang) + (fix(-diff(tgtang)/4)).*2.*pi;
  dc = -sqrt(diff(vtx).*diff(vtx) + diff(vty).*diff(vty));%this is negative because of direction of curve traverse
  curv = dtheta./dc;
  theta = atan2(vty, vtx) + lt(atan2(vty, vtx), 0.0).*2.*pi;
  rad = sqrt(vtx.*vtx + vty.*vty);
  rvx = rad.*cos(theta);  
  rvy = rad.*sin(theta);
  cmtheta = atan2(com(1,2), com(1, 1));
  cmr = sqrt(com(1,1).*com(1,1) + com(1,2).*com(1,2));
  cmx = cmr.*cos(cmtheta);
  cmy = cmr.*sin(cmtheta);
  ang = tgtang - 2.*pi.*fix((tgtang)./(2*pi));
  ap = atan2((rvy - cmy), (rvx - cmx)) + lt(atan2((rvy - cmy), (rvx - cmx)), 0.0).*2.*pi;
  angpos = ap(1, 2:size(ap,2));
  