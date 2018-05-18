clearvars;

% clearvars;
dinfo = dir('../Out/rgb/');
fid = fopen('../Out/rgb.txt','w');
for i = 1:size(dinfo)
    if(dinfo(i).bytes > 0)
        fprintf(fid,'%s\n',dinfo(i).name);
    end
end
fclose(fid);
command = './extract_focal.pl';
system(command);
copyfile('list.txt','../Out/list_focal.txt');

a = textread('config.txt','%s');
%exePath = '/home/tcsuser/Desktop/SFM/SiftGPU/';
exePath = strcat(a,'SiftGPU/');

%outPath = '/home/tcsuser/Desktop/SFM/Out/';

outPath = strcat(a,'Out/');;

% ./SiftGPU/SiftExtractor ./Out/ ./Out/
command = strcat(exePath, '/SiftExtractor /');
command = strcat(strcat(command, outPath), ' /');
command = strcat(command, outPath)
status = system(char(command));

%./SiftGPU/VocabLearn ./Out/KeyList.txt 4 8 1 ./Out/tree.out
fileName = strcat(outPath, 'sift_count.txt');
nSift = load(char(fileName));
n = nSift/100;
b = 9;
h = log10(n)/log10(9);
h = int16(h + 0.5);
a = textread('config.txt','%s');

%exePath = '/home/tcsuser/Desktop/SFM/SiftGPU/';
exePath = strcat(a,'SiftGPU/');

%outPath = '/home/tcsuser/Desktop/SFM/Out/';

outPath = strcat(a,'Out/');;

command = strcat(exePath, '/VocabLearn /');
command = strcat(strcat(command, outPath), '/KeyList1.txt ');
c = sprintf(' %d %d 1 /', h, b);
command = strcat(command, c);
command = strcat(strcat(command, outPath), '/tree.out')
status = system(char(command));

%./SiftGPU/VocabBuildDB ./Out/KeyList.txt ./Out/tree.out ./Out/db_tree.out
a = textread('config.txt','%s');
%exePath = '/home/tcsuser/Desktop/SFM/SiftGPU/';
exePath = strcat(a,'SiftGPU/');

%outPath = '/home/tcsuser/Desktop/SFM/Out/';

outPath = strcat(a,'Out/');;

command = strcat(exePath, '/VocabBuildDB /');
command = strcat(strcat(command, outPath), '/KeyList1.txt /');
command = strcat(strcat(command, outPath), '/tree.out /');
command = strcat(strcat(command, outPath), '/db_tree.out')
status = system(char(command));

% ./SiftGPU/VocabMatch ./Out/db_tree.out ./Out/KeyList.txt ./Out/KeyList.txt 80 ./Out/matches.txt
a = textread('config.txt','%s');
%exePath = '/home/tcsuser/Desktop/SFM/SiftGPU/';
exePath = strcat(a,'SiftGPU/');

%outPath = '/home/tcsuser/Desktop/SFM/Out/';

outPath = strcat(a,'Out/');;

command = strcat(exePath, '/VocabMatch /');
command = strcat(strcat(command, outPath), '/db_tree.out /');
command = strcat(strcat(command, outPath), '/KeyList1.txt /');
command = strcat(strcat(command, outPath), '/KeyList1.txt 80 /');
command = strcat(strcat(command, outPath), '/matches.txt')
status = system(char(command));

% ./SiftGPU/MatchSift ./Out/KeyList.txt ./Out/matches.txt ./Out/
a = textread('config.txt','%s');
%exePath = '/home/tcsuser/Desktop/SFM/SiftGPU/';
exePath = strcat(a,'SiftGPU/');

%outPath = '/home/tcsuser/Desktop/SFM/Out/';

outPath = strcat(a,'Out/');;

command = strcat(exePath, '/MatchSift /');
command = strcat(strcat(command, outPath), '/matches.txt /');
command = strcat(command, outPath);
status = system(char(command));

%dendogram
a = textread('config.txt','%s');
%exePath = '/home/tcsuser/Desktop/SFM/SiftGPU/';
exePath = strcat(a,'SiftGPU/');

%outPath = '/home/tcsuser/Desktop/SFM/Out/';

outPath = strcat(a,'Out/');;

dendrogram_exp

%./SiftGPU/ClusterReconstruct ./Out/
a = textread('config.txt','%s');
%exePath = '/home/tcsuser/Desktop/SFM/SiftGPU/';

exePath = strcat(a,'SiftGPU/');

%outPath = '/home/tcsuser/Desktop/SFM/Out/';

outPath = strcat(a,'Out/');;
command = strcat(exePath, '/ClusterReconstruct /');
command = strcat(command, outPath)
status = system(char(command));

% R, C estimation
main

% Triangulate
a = textread('config.txt','%s');
%exePath = '/home/tcsuser/Desktop/SFM/SiftGPU/';
exePath = strcat(a,'SiftGPU/');

%outPath = '/home/tcsuser/Desktop/SFM/Out/';

outPath = strcat(a,'Out/');;

command = strcat(exePath, '/TriangulateMulti /');
command = strcat(command, outPath)
command = strcat('LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tcsuser/ORB_SLAM2_CV_3/Thirdparty/g2o/lib/;', command);
status = system(char(command));

%cluster merging
main_rot_avg
