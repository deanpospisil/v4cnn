%lets get all the ndata in a directory
cd('/Users/dean/Desktop/modules/v4cnn/data/responses/apc_orig_resp/')
s = dir(pwd); % s is structure array with fields name, 
                   % date, bytes, isdir
file_list = {s.name}'; % convert the name field from the elements
                        % of the structure array into a cell array
                        % of strings.
                        
apc_files = [];
for file = 1:length(file_list)
    if isempty(strfind(file_list{file}, '.'))
        %apc_files = cat(1, apc_files, file_list(file));
        apc_files = cat(1, nd_read())
    end
end
