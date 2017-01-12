%
%  nd_read.m
%  Wyeth Bair
%  Dec 21, 2014
%
%  The matlab functions below will read an nData file (binary format) into
%  a set of struct objects.
%
%  For documentation, go to:  www.iModel.org/nd/  and click on the
%    "MATLAB format" link on the left.
%
%  NOTES
%    (1) Endian is not checked.
%    (2) Event code table is not handled.
%
%**************************************-**************************************%
%                                                                             %
%                                   ND_READ                                   %
%                                                                             %
%  Read an nData file.                                                        %
%                                                                             %
%*****************************************************************************%
function [ nd ] = nd_read(infile)

revflag = 0;

fin = fopen(infile);

%  Read the first [int], if it is not 1, change endian.
tval = fread(fin,1,'int');
if (tval ~= 1)
  fprintf('  *** Cannot read file - possible endian problem.  Exiting\n');
  return;
end

%
%  Read the file class, which is a string describing the file
%
nd.class = ndata_read_nchar(fin,revflag);
fprintf('    File class:  %s\n',nd.class);

%
%  Read the constant parameters
%
nd.nconst = fread(fin,1,'int');  % Number of constant parameters
fprintf('    Constant parameters:  %d\n',nd.nconst);
for i=1:nd.nconst
  nd.const(i).name = ndata_read_nchar(fin,revflag);
  nd.const(i).type = char(fread(fin,1,'char'));
  nd.const(i).val  = ndata_read_nchar(fin,revflag);
end

%
%  Read names and types of variable parameters
%
nd.nvar = fread(fin,1,'int');  % Number of variable parameters
fprintf('    Variable parameters:  %d\n',nd.nvar);
if (nd.nvar > 0)
  fprintf('     ');
end
for i=1:nd.nvar
  nd.var(i).name = ndata_read_nchar(fin,revflag);
  nd.var(i).type = char(fread(fin,1,'char'));
  fprintf(' %s',nd.var(i).name);
end
fprintf('\n');

%
%  Read the event code table
%
nd.ntable = fread(fin,1,'int');  % Number of event-code tables
if nd.ntable > 0
  fprintf('  *** This feature not available:  event code tables\n');
  return;
end

%
%  Read the number of trials
%
nd.ntrial = fread(fin,1,'int');  % Number of trials
fprintf('    Trials:  %d\n',nd.ntrial);

for i=1:nd.ntrial
  %
  %  Read the header information for this trial
  %
  nd.tr(i).tcode  = fread(fin,1,'int');
  nd.tr(i).tref   = fread(fin,1,'int');
  nd.tr(i).nparam = fread(fin,1,'int');   % Number of variable params
  for j=1:nd.tr(i).nparam
    nd.tr(i).par(j).name = ndata_read_nchar(fin,revflag);
    nd.tr(i).par(j).val  = ndata_read_nchar(fin,revflag);
  end

  nd.tr(i).nrec = fread(fin,1,'int');   % Number of records (aka channels)
  for j=1:nd.tr(i).nrec
    nd.tr(i).r(j).rtype = fread(fin,1,'int');
    nd.tr(i).r(j).name  = ndata_read_nchar(fin,revflag);
    nd.tr(i).r(j).rcode = fread(fin,1,'int');
    nd.tr(i).r(j).sampling = fread(fin,1,'float');
    nd.tr(i).r(j).t0    = fread(fin,1,'int');
    nd.tr(i).r(j).tn    = fread(fin,1,'int');
    nd.tr(i).r(j).n     = fread(fin,1,'int');

    if nd.tr(i).r(j).rtype == 0
      nd.tr(i).r(j).p = fread(fin,nd.tr(i).r(j).n,'int');
    elseif nd.tr(i).r(j).rtype == 1
      nd.tr(i).r(j).x = fread(fin,nd.tr(i).r(j).tn,'float');
    elseif nd.tr(i).r(j).rtype == 2
      for k=1:nd.tr(i).r(j).n
        nd.tr(i).r(j).p(k) = fread(fin,1,'int');
        nd.tr(i).r(j).x(k) = fread(fin,1,'float');
      end
    else
      fprintf('  *** Feature not implemented yet:  rtype > 2\n');
      return;
    end
  end  
end
fclose(fin);

end
%**************************************-**************************************%
%                                                                             %
%                               NDATA_READ_NCHAR                              %
%                                                                             %
%*****************************************************************************%
function [ s ] = ndata_read_nchar(fin,revflag)

  nc = fread(fin,1,'int');  % Number of chars to read
  if nc == 0
    fprintf('  *** Expecting value > 0\n');
    return;
  end

  sa = fread(fin,nc,'char');  % This is an array of byte values
  s = char(sa.');  % Make a string from the transposed array
end