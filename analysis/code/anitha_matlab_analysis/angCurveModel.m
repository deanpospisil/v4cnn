% Samplez file for a two dimensional gaussian fit of data using nonlin least
% squares.
%other files you need stimdimt (text file), maxmodel, maxres
%you need to set the activity of a neuron to the 370 shapes to
%resp(:,2)
clear all
[top_dir, temp] = strsplit(pwd, '/analysis/');
top_dir = top_dir{1};
respDir= [top_dir '/data/responses/'];
fitDir= [top_dir 'data/an_results'];

source='V4';
stim='370PC2001';
% source= 'v41cellDina';
% stim='370PC2001';

fitting='LSQnonlin';

respMatFile=[respDir source '_' stim '.mat']
fitOutFile=[fitDir source '_' stim  '_' fitting '.mat']

a=load(respMatFile);
sourceResp=a.resp;


global resp fe err dim db
saveIter=0;
nullHyp=false;
unit=1;
nParams=5;
for layer=1:length(sourceResp)
    bestfitparams{layer}=nan(size(sourceResp{layer},1),nParams);
end
% 
% %fitting the layers
% for layer=1:length(sourceResp)%go through each layer
%     
%         for unit=1:size(sourceResp{layer},1)
%            
%             layer
%             unit
%             
%             if sum(sourceResp{layer}(unit,:))>0 && sum(sourceResp{layer}(unit,:)>0)>0
%                 temp = sourceResp{layer}(unit,:); % Set this to the responses of the 370 shapes.
%                 if nullHyp
%                     resp(:,2) = temp(randperm(length(temp)));%shuffle responses for null hyp    
%                 else
%                     resp(:,2) = temp;  
%                 end
%                 
%                 dim = 2;
%                 fidr = fopen('stimdimt', 'r'); % stimulus representation file
%                 tstim = fscanf(fidr, '%f', [8, 2298]);
%                 stim = transpose(tstim);
%                 resp(:,1) = fscanf(fidr, '%f', [370, 1]); % number of points that represent each stimulus
%                 fe(:,1:2) = stim(:, 1:2); % just taking the angular position and curvature dimensions
%                 fclose(fidr);
%                 options = optimset('diffmaxchange',0.5,'maxfunevals', 50000, 'maxiter', 20000, 'tolX', 0.0001, 'tolfun', 0.0001);
%                 
%                 ub = ([7.0 1.0 inf
%                     inf inf inf]);
%                 lb = ([-0.7 -1.0 -inf
%                     0.0 0.0 0.0]);
%                 X0=zeros(20,2,3);
%                 
%                 curve=[ 1 0.5 0 -0.5 -1];
%                 or=[0 pi/2 pi 3*pi/2];
%                 [x y]=meshgrid(curve,or);
%                 
%                 curor=zeros(2,20);
%                 curor(1,:)=reshape(x, 1, length(or)*length(curve));
%                 curor(2,:)=reshape(y, 1, length(or)*length(curve));
%                 
%                 
%                 for j = 1:length(curor)
%                     
%                     unit
%                     X0 = ([curor(2,j) curor(1,j) 1.0; 10 10 0.0])
%                     % starting point for the least square minimization, try multiple starting points
%                     %The first column is the mean and 1/sd of the ang pos dimension, second
%                     %column is the mean and 1/sd of the curv dim and the third column, top
%                     %row is the constant
%                     
%                     [X, cc, cp] = maxmodel(X0, lb, ub, options);
%                     disp(cc);
%                     max11(j,:) = [X(1, :) X(2,:) cc(1,2)]; % this is the fitted params
%                     maxint11(j, :) = [cp(:,1)' cp(:,2)']; % this is the confidence interval
%                     correlations(j)=cc(1,2);
%                     potErr(j,:,:)=err;%first column of err, is the actual response
%                     
%                 end
%                 
%                 [~, bestfitid] = max(max11(:,7));
%                 
%                 %here we get a unit added to best fit params
%                 bestfitparams{layer}(unit,:) = [max11(bestfitid,[1:2 4:5 7])]; % params order originally:mean angpos (radians), mean curv, const,1/sd angpos, 1/sd cur, correlation
%                 bestfitparams{layer}(unit,3:4) = bestfitparams{layer}(unit,3:4).^-1;%transform to SD 
%                 saveIter=saveIter+1;
%             else
%                 bestfitparams{layer}(unit,:) = nan;
%              
%             end
%             
%             if saveIter>50 || layer==size(sourceResp,1) & unit==size(sourceResp{end},1)
%                 %get into the correct format
%                 
%                 fI=bestfitparams;
%  
%                
%                 if nullHyp
% %                     save([fitOutFile '_ScrambledResp'],'fI')
%                 else
%                       save([fitOutFile ],'fI')
%                 end
%                 saveIter=0;
%             end
%             
%         end
%     end
% 
% 
