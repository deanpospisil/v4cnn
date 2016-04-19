load(['/home/dean/Desktop/AlexNetResults/pasCenterResults/untrained' num2str(1) '.mat'])
numStim=370;

%featureXimg make the array to hold each units response to all stimuli
fI=cell(size(resp));
for ind=1:length(fI)
    
    fI{ind}=zeros( length(resp{ind}), numStim);
end

for imNum=1:numStim
    
    load(['/home/dean/Desktop/AlexNetResults/pasCenterResults/' num2str(imNum) '.mat']);
    
    imNum/numStim
    for ind=1:length(fI)%go through each layer
        
        mid=round(size(resp{ind},2)/2);%get the middle of each conv layer
        
        for featureNum=1:length(resp{ind})%go through each feature in each layer
            
            if numel(size(resp{ind}))>2%for conv layers just pull out the middle features
                fI{ind}(featureNum,imNum) = resp{ind}(featureNum,mid,mid);
            else%for fully connected just get all the features
                fI{ind}(featureNum,imNum) = resp{ind}(featureNum);
            end
            
        end
        
    end
    

end

save('/home/dean/Desktop/untrainedRespPCorig','fI');