function [X, cc, cp,residual] = maxmodel(X0, lb, ub, options)

global resp fe err dim
[X, resnorm, residual, exitflag,out,lam,jay] = lsqnonlin('maxres', X0, lb, ub, options);
disp(X);
p = [X(1,:) X(2,:)];
cc = corrcoef(err);
cint = nlparci(p, residual, jay);
cp = cint(1:(dim*2+1),:);
