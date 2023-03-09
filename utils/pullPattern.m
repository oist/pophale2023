function currImg = pullPattern(dataFolder, suffix,ht,currDset,indRelDset)

allFiles = dir([dataFolder '/*' suffix]);
af=[];
embedFiles=[];
for x=1:numel(allFiles)
    af{x}=[allFiles(x).folder '/' allFiles(x).name];
    basename=split(allFiles(x).name,'_');
    embedFiles{x}=[allFiles(x).folder '/' basename{1}];
end
allFiles=af;

fileInfo=h5info(embedFiles{1},'/patterns1');
embedSize=fileInfo.Dataspace.Size;


if ht(currDset)==1
    try
        currImg=h5read(embedFiles{currDset}, '/patterns1',[1,1,1,indRelDset],[embedSize(1),embedSize(2),embedSize(3),1]);
    catch
         currImg=h5read(embedFiles{currDset}, '/patterns1',[1,1,1,indRelDset-1],[embedSize(1),embedSize(2),embedSize(3),1]);
        
    end
else
    try
        currImg=h5read(embedFiles{currDset}, '/patterns2',[1,1,1,indRelDset],[embedSize(1),embedSize(2),embedSize(3),1]);
    catch
         currImg=h5read(embedFiles{currDset}, '/patterns2',[1,1,1,indRelDset-1],[embedSize(1),embedSize(2),embedSize(3),1]);
    end
end
currImg=permute(currImg,[3,2,1]);